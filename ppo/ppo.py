
from .actor_critic import ActorCritic
from .dataset import RecurrentPPODataset

from typing import Union
import numpy as np
import torch
from torch.utils.data import TensorDataset, BatchSampler, RandomSampler


class _RecurrentPPO:
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
        num_traj_minibatch: int,
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
        min_seq_size: Union[int, None], # not used when None
        device: str):
        """
        num_steps: number of steps to run in each environment per policy rollout
        num_envs: number of parallel game environments
        observation_size: shape of input to actor/critic
        action_size: shape of output of actor
        learning_rate: learning rate of the optimizer
        num_traj_minibatch: number of trajectories in each minibatch during update
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
        min_seq_size: minimum length of sequence during training, all shorter sequences are discarted.
            (only used for recurrent PPO).
        """

        self._actor_critic = actor_critic
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._obs_size_actor = obs_size_actor
        self._obs_size_critic = obs_size_critic
        self._action_size = action_size
        self._learning_rate = learning_rate
        self._num_traj_minibatch = num_traj_minibatch
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
        self._min_seq_size = min_seq_size
        self._device = device

        self.RECURRENT = None

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

    def act(self, obs_actor, obs_critic, actor_state, critic_state):
        with torch.no_grad():
            actions, logprobs, _, actor_state = self._actor_critic.get_action(obs_actor, actor_state)
            values, critic_state = self._actor_critic.get_value(obs_critic, critic_state)
            
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
                critic_state)

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

    def _update_step(self, newlogprob, logprobs, advantages, returns, newvalue, values, entropy):

        logratio = newlogprob - logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            # clipfracs += [((ratio - 1.0).abs() > self._clip_coef).float().mean().item()]

        if self._use_norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1 - self._clip_coef, 1 + self._clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)
        if self._clip_vloss:
            v_loss_unclipped = (newvalue - returns) ** 2
            v_clipped = values + torch.clamp(
                newvalue - values,
                -self._clip_coef,
                self._clip_coef,
            )
            v_loss_clipped = (v_clipped - returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self._ent_coef * entropy_loss + v_loss * self._vf_coef

        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._actor_critic.parameters(), self._max_grad_norm)
        self._optimizer.step()

        return v_loss, pg_loss, entropy_loss, approx_kl


    def split_sequences(self, obs, dones, actor_states, critic_states, tensors):
        """
        Splits a batch of sequences containing several trajectories into several
        sequences that contain only one trajectory.
        This is accomplished by splitting the input sequence everywhere where
        there's a done.

        Original sequences: [[a1, a2, a3, a4 | a5, a6],
                             [b1, b2 | b3, b4, b5 | b6]]

        Split sequences:    [[a1, a2, a3, a4],
                             [a5, a6],
                             [b1, b2],
                             [b3, b4, b5],
                             [b6]]

        These split sequences can be then joined together into a batch by
        padding them. This enables a computationally efficient way to process
        entire batches of sequences, without having to iterate over the
        individual steps.

        Attention: this method can lead to very large batch sizes. This becomes
        a problem if input contains many small trajectories. For this reason,
        trajectories shorter than `_min_seq_size` are removed.
        """

        # input must be (seq_len, batch_size, obs_size)
        assert obs.ndim == 3
        seq_len = obs.shape[0]
        batch_size = obs.shape[1]

        # SPLIT TRAJECTORIES
        # clone, as this tensor will be modified
        split_indices = dones.clone()
        # force split at the start of trajectory
        split_indices[0] = 1
        # transpose and flatten tensors to reorder entries
        # required later for splitting the observations correctly
        # for tensor with indices (sequence, batch):
        #   old shape: [obs(0,0), obs(0,1), ...]
        #   new shape: [obs(0,0), obs(1,0), ...]
        # causes data to be copied
        obs = obs.movedim(0, 1).flatten(0, 1)
        dones = dones.movedim(0, 1).flatten(0, 1)
        split_indices = split_indices.movedim(0, 1).flatten(0, 1)
        # get indices (split requires tensor in CPU)
        split_indices = torch.argwhere(split_indices).flatten().cpu()
        # split trajectories at dones or start of trajectory
        obs_split = list(torch.tensor_split(obs, split_indices, dim=0))
        # remove first element if empty
        if obs_split[0].numel() == 0:
            obs_split = obs_split[1:]

        # REMOVE SHORT TRAJECTORIES
        # get start and end of each trajectory
        # for this we need to add the end of last trajectory to split_indices
        start_end_indices = torch.cat((split_indices, torch.tensor([batch_size*seq_len])))
        # get original length of each trajectory
        traj_lengths = torch.diff(start_end_indices)
        # select trajectories to remove (smaller than x steps)
        remove_mask = traj_lengths < self._min_seq_size
        remove_indices = torch.argwhere(remove_mask).flatten()
        keep_indices = torch.argwhere(torch.logical_not(remove_mask)).flatten()
        # update trajectory lengths
        traj_lengths = traj_lengths[keep_indices]
        # filter trajectories only if needed
        if torch.any(remove_mask):
            # reverse indices otherwise only the first index is correctly removed
            for index in reversed(remove_indices):
                del obs_split[index]
            assert len(obs_split) > 0

        # OTHER TENSORS
        # apply same operations as above to the other tensors
        tensors_split = []
        for t in tensors:
            t = t.movedim(0, 1).flatten(0, 1)
            t_split = list(torch.tensor_split(t, split_indices, dim=0))
            if t_split[0].numel() == 0:
                t_split = t_split[1:]
            if torch.any(remove_mask):
                # reverse indices otherwise only the first index is correctly removed
                for index in reversed(remove_indices):
                    del t_split[index]
                assert len(t_split) > 0
            tensors_split.append(t_split)

        # GENERATE HIDDEN STATES
        # get indices of started trajectories
        # (the ones that have an initial hidden state)
        started_traj_indices = seq_len*torch.arange(batch_size)
        # batches corresponding to started trajectories
        started_traj_batches = torch.argwhere(torch.isin(start_end_indices, started_traj_indices)).flatten()
        # generate mask for started trajectories that have a done
        # (these should be discarded)
        started_done_traj_mask = torch.logical_not(dones[started_traj_indices] > 0.5).cpu()
        # generate mask for new hidden states only with relevant batches
        new_batches_mask = torch.isin(keep_indices, started_traj_batches[started_done_traj_mask])
        # generate mask for hidden states of started trajectories
        started_batches_mask = torch.isin(started_traj_batches, keep_indices)
        started_batches_mask = torch.logical_and(started_batches_mask, started_done_traj_mask)
        # insert existing hidden states of started trajectories
        h_actor = self._actor_critic.init_hidden(len(obs_split))
        h_actor[..., new_batches_mask, :] = actor_states[..., started_batches_mask, :]
        h_critic = self._actor_critic.init_hidden(len(obs_split))
        h_critic[..., new_batches_mask, :] = critic_states[..., started_batches_mask, :]

        # GENERATE MASK
        # get mask of valid trajectories
        traj_mask = traj_lengths > torch.arange(0, seq_len).unsqueeze(1)

        print("Num trajectories before/after split: ", batch_size, "/", len(obs_split))

        return obs_split, traj_mask, h_actor, h_critic, tensors_split

    def apply_mask(self, tensors, mask):
        """
        Joins tensors together by selecting only unpadded steps.
        """
        assert isinstance(tensors, list)
        tensors_full = []
        for t in tensors:
            # create array with full size of output
            # (as if short trajectories were not removed)
            t_full = torch.zeros(*mask.shape, device=t.device)
            # flatten dimensions and move data
            t_full[:t.shape[0]] = t.flatten(1, -1)
            # apply mask to unpad outputs and join them
            t_full = t_full.movedim(0, 1)[mask.movedim(0, 1)]
            tensors_full.append(t_full)

        return tensors_full


class RecurrentPPO(_RecurrentPPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RECURRENT = True
        assert self._actor_critic._recurrent_actor == True or self._actor_critic._recurrent_critic == True

    def update(self, initial_actor_state, initial_critic_state):

        # Preprocess collected data, create a dataset and specify sampling strategy
        # Recurrent approach:
        # 1. Split obs and other collected data into lists of single trajectories
        #    (split sequences everywhere there's a done)
        # 2. Select random trajectories and build batches of size `num_traj_minibatch`
        # 3. Get a batch of trajectories and join them by padding the trajectories
        # 4. After inference, apply mask to output to select only valid (non-padded) steps
        # Benefits of this approach:
        # - Faster inference, as we don't have to iterate over individual steps
        # - Memory footprint is limited to tensor of size `num_traj_minibatch*num_steps`
        #   (upper bound, as collected trajectories can be smaller)
        obs_actor, traj_mask, initial_actor_state, initial_critic_state,\
        [obs_critic, actions, logprobs, advantages, returns, values] \
            = self.split_sequences(
                self._obs_actor, self._dones, initial_actor_state, initial_critic_state,
                [self._obs_critic, self._actions, self._logprobs, self._advantages, self._returns, self._values])

        # has to be the same order as below
        dataset = RecurrentPPODataset(obs_actor, obs_critic, actions, logprobs, advantages, returns, values)
        sampler = BatchSampler(RandomSampler(dataset), self._num_traj_minibatch, False)

        v_losses = 0.
        pg_losses = 0.
        entropy_losses = 0.

        for epoch in range(self._update_epochs):
            for batch_indices in sampler:

                # get batch data
                # has to have same order as above
                obs_actor_b, obs_critic_b, actions_b, logprobs_b, advantages_b, returns_b, values_b \
                    = dataset[batch_indices]
                actor_state = initial_actor_state[..., batch_indices, :] if torch.is_tensor(initial_critic_state) else None
                critic_state = initial_critic_state[..., batch_indices, :] if torch.is_tensor(initial_critic_state) else None

                _, newlogprob, entropy, _ = self._actor_critic.get_action(
                    obs=obs_actor_b,
                    hidden_states=actor_state,
                    action=actions_b)

                newvalue, _ = self._actor_critic.get_value(
                    obs=obs_critic_b,
                    hidden_states=critic_state)

                # apply mask to select only unpadded data
                [newlogprob, logprobs_b, advantages_b, newvalue, returns_b, values_b, entropy] \
                    = self.apply_mask(
                        [newlogprob, logprobs_b, advantages_b, newvalue, returns_b, values_b, entropy],
                        traj_mask[:, batch_indices])

                v_loss, pg_loss, entropy_loss, approx_kl = self._update_step(
                    newlogprob=newlogprob,
                    logprobs=logprobs_b,
                    advantages=advantages_b,
                    returns=returns_b,
                    newvalue=newvalue,
                    values=values_b,
                    entropy=entropy)

                v_losses += v_loss.item()
                pg_losses += pg_loss.item()
                entropy_losses += entropy_loss.item()

            if self._target_kl is not None and approx_kl > self._target_kl:
                break

        # reset storage
        self._init_storage()
        self._scheduler.step()

        return v_losses, pg_losses, entropy_losses

class PPO(_RecurrentPPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RECURRENT = False
        assert self._actor_critic._recurrent_actor == False
        assert self._actor_critic._recurrent_critic == False

    def act(self, obs_actor, obs_critic):
        actions, logprobs, values, _, _ = super().act(
            obs_actor=obs_actor,
            obs_critic=obs_critic,
            actor_state=None,
            critic_state=None)
        return actions, logprobs, values

    def compute_returns(self, obs_critic, dones):
        return super().compute_returns(
            obs_critic=obs_critic,
            critic_state=None,
            dones=dones)

    def update(self):

        # Non-recurrent approach:
        # 1. Reshape experience into batch of individual steps
        # 2. Select batches of steps
        # flatten the data
        obs_actor = self._obs_actor.reshape((-1, self._obs_size_actor))
        obs_critic = self._obs_critic.reshape((-1, self._obs_size_critic))
        actions = self._actions.reshape((-1, self._action_size))
        logprobs = self._logprobs.reshape(-1)
        dones = self._dones.reshape(-1)
        advantages = self._advantages.reshape(-1)
        returns = self._returns.reshape(-1)
        values = self._values.reshape(-1)

        # has to be the same order as below
        dataset = TensorDataset(obs_actor, obs_critic, actions, logprobs, advantages, returns, values)
        sampler = BatchSampler(RandomSampler(dataset), self._num_traj_minibatch*self._num_steps, False)

        v_losses = 0.
        pg_losses = 0.
        entropy_losses = 0.

        for epoch in range(self._update_epochs):
            for batch_indices in sampler:

                # get batch data
                # has to have same order as above
                obs_actor_b, obs_critic_b, actions_b, logprobs_b, advantages_b, returns_b, values_b \
                    = dataset[batch_indices]

                _, newlogprob, entropy, _ = self._actor_critic.get_action(
                    obs=obs_actor_b,
                    hidden_states=None,
                    action=actions_b)

                newvalue, _ = self._actor_critic.get_value(
                    obs=obs_critic_b,
                    hidden_states=None)

                v_loss, pg_loss, entropy_loss, approx_kl = self._update_step(
                    newlogprob=newlogprob,
                    logprobs=logprobs_b,
                    advantages=advantages_b,
                    returns=returns_b,
                    newvalue=newvalue,
                    values=values_b,
                    entropy=entropy)

                v_losses += v_loss.item()
                pg_losses += pg_loss.item()
                entropy_losses += entropy_loss.item()

            if self._target_kl is not None and approx_kl > self._target_kl:
                break

        # reset storage
        self._init_storage()
        self._scheduler.step()

        return v_losses, pg_losses, entropy_losses
