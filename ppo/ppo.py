
from .actor_critic import ActorCritic

from typing import Union
import numpy as np
import torch

class PPO:
    """
    Implementation of PPO based on:
    https://github.com/vwxyzjn/cleanrl
    https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """

    def __init__(self, 
        actor_critic: ActorCritic,
        num_steps: int,
        num_envs: int,
        obs_size_actor: int,
        obs_size_critic: int,
        action_size: int,
        learning_rate: float,
        num_minibatches: int,
        update_epochs: int,
        use_gae: bool,
        gae_lambda: float,
        gamma: float,
        clip_vloss: bool,
        clip_coef: float,
        ent_coef: float,
        vf_coef: float,
        use_norm_adv: bool,
        max_grad_norm: float,
        target_kl: Union[float, None], # not used when None
        learning_rate_gamma: Union[float, None], # not used when None
        device: str):
        """
        num_steps: number of steps to run in each environment per policy rollout
        num_envs: number of parallel game environments
        observation_size: shape of input to actor/critic
        action_size: shape of output of actor
        learning_rate: learning rate of the optimizer
        num_minibatches: number of mini-batches
        update_epochs: number of epochs to update the policy
        use_gae: use GAE for advantage computation
        gae_lambda: lambda for the general advantage estimation
        gamma: discount factor gamma
        clip_vloss: use a clipped loss for the value function
        clip_coef: surrogate clipping coefficient
        ent_coef: entropy coefficient
        vf_coef: value function coefficient
        use_norm_adv: use advantages normalization
        max_grad_norm: maximum norm for the gradient clipping
        target_kl: if not None, target KL divergence threshold
        learning_rate_gamma: if not None, multiplicative factor of learning rate decay
        """

        self._actor_critic = actor_critic
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._obs_size_actor = obs_size_actor
        self._obs_size_critic = obs_size_critic
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._num_minibatches = num_minibatches
        self._update_epochs = update_epochs
        self._use_gae = use_gae
        self._gae_lambda = gae_lambda
        self._gamma = gamma
        self._clip_vloss = clip_vloss
        self._clip_coef = clip_coef
        self._ent_coef = ent_coef
        self._vf_coef = vf_coef
        self._use_norm_adv = use_norm_adv
        self._max_grad_norm = max_grad_norm
        self._target_kl = target_kl
        self._device = device

        # batch size for flat trajectories
        self._batch_size = int(self._num_envs * self._num_steps)

        # optimizer
        self._optimizer = torch.optim.Adam(self._actor_critic.parameters(), lr=self._learning_rate, eps=1e-5)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, learning_rate_gamma, last_epoch=-1, verbose=True)

        self._init_storage()

    def _init_storage(self):
        # storage setup
        self._obs_actor = torch.zeros((self._num_steps, self._num_envs, self._obs_size_actor)).to(self._device)
        self._obs_critic = torch.zeros((self._num_steps, self._num_envs, self._obs_size_critic)).to(self._device)
        self._actions = torch.zeros((self._num_steps, self._num_envs, self._action_size)).to(self._device)
        self._logprobs = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
        self._rewards = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
        self._dones = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
        self._values = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
        self._advantages = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
        self._returns = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
        self._step = 0

    def act(self, obs_actor, obs_critic, actor_state, critic_state, dones):
        with torch.no_grad():
            actions, logprobs, _, actor_state = self._actor_critic.get_action(obs_actor, actor_state, dones)
            values, critic_state = self._actor_critic.get_value(obs_critic, critic_state, dones)
            
        return actions, logprobs, values, actor_state, critic_state

    def set_step(self, obs_actor, obs_critic, actions, logprobs, rewards, dones, values):
        self._obs_actor[self._step] = obs_actor
        self._obs_critic[self._step] = obs_critic
        self._actions[self._step] = actions
        self._logprobs[self._step] = logprobs
        self._rewards[self._step] = rewards
        self._dones[self._step] = dones
        self._values[self._step] = values.flatten()
        # increment step
        self._step += 1

    def compute_returns(self, obs_critic, critic_state, dones):
        # bootstrap value if not done
        with torch.no_grad():
            value, _ = self._actor_critic.get_value(
                obs_critic,
                critic_state,
                dones,
            )
            value = value.reshape(1, -1)
            if self._use_gae:
                self._advantages = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
                lastgaelam = 0
                for t in reversed(range(self._num_steps)):
                    if t == self._num_steps - 1:
                        nextnonterminal = 1.0 - dones.float()
                        nextvalues = value
                    else:
                        nextnonterminal = 1.0 - self._dones[t + 1]
                        nextvalues = self._values[t + 1]
                    delta = self._rewards[t] + self._gamma * nextvalues * nextnonterminal - self._values[t]
                    self._advantages[t] = lastgaelam = delta + self._gamma * self._gae_lambda * nextnonterminal * lastgaelam
                self._returns = self._advantages + self._values
            else:
                self._returns = torch.zeros((self._num_steps, self._num_envs)).to(self._device)
                for t in reversed(range(self._num_steps)):
                    if t == self._num_steps - 1:
                        nextnonterminal = 1.0 - dones.float()
                        next_return = value
                    else:
                        nextnonterminal = 1.0 - self._dones[t + 1]
                        next_return = self._returns[t + 1]
                    self._returns[t] = self._rewards[t] + self._gamma * nextnonterminal * next_return
                self._advantages = self._returns - self._values

    def update(self, initial_actor_state: torch.Tensor, initial_critic_state: Union[None, torch.Tensor]):

        # flatten the data
        obs_actor_flat = self._obs_actor.reshape((-1, self._obs_size_actor))
        obs__critic_flat = self._obs_critic.reshape((-1, self._obs_size_critic))
        logprobs_flat = self._logprobs.reshape(-1)
        actions_flat = self._actions.reshape((-1, self._action_size))
        dones_flat = self._dones.reshape(-1)
        advantages_flat = self._advantages.reshape(-1)
        returns_flat = self._returns.reshape(-1)
        values_flat = self._values.reshape(-1)

        # get batch indices
        assert self._num_envs % self._num_minibatches == 0
        envsperbatch = self._num_envs // self._num_minibatches
        envinds = np.arange(self._num_envs)
        flatinds = np.arange(self._batch_size).reshape(self._num_steps, self._num_envs)
        clipfracs = []

        # Optimizing the policy and value network
        for epoch in range(self._update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, self._num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, _ = self._actor_critic.get_action(
                    obs_actor_flat[mb_inds],
                    initial_actor_state[..., mbenvinds, :],
                    dones_flat[mb_inds],
                    actions_flat[mb_inds],
                )
                newvalue, _ = self._actor_critic.get_value(
                    obs__critic_flat[mb_inds],
                    initial_critic_state[..., mbenvinds, :] if torch.is_tensor(initial_critic_state) else None,
                    dones_flat[mb_inds],
                )
                logratio = newlogprob - logprobs_flat[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self._clip_coef).float().mean().item()]

                mb_advantages = advantages_flat[mb_inds]
                if self._use_norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if self._clip_vloss:
                    v_loss_unclipped = (newvalue - returns_flat[mb_inds]) ** 2
                    v_clipped = values_flat[mb_inds] + torch.clamp(
                        newvalue - values_flat[mb_inds],
                        -self._clip_coef,
                        self._clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns_flat[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns_flat[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self._ent_coef * entropy_loss + v_loss * self._vf_coef

                self._optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self._max_grad_norm)
                self._optimizer.step()

            if self._target_kl is not None and approx_kl > self._target_kl:
                break

        # reset storage
        self._init_storage()
        self._scheduler.step()

        return v_loss, pg_loss, entropy_loss
