
import random

import gym
import numpy as np
import torch

from ppo import PPO, ActorCritic


class RNN(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            device: str):
        super().__init__()

        activation = torch.nn.LeakyReLU
        hidden_size = 64
        self._rnn_hidden_size = 5
        self._num_layers = 2
        self._device = device

        self._mlp1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, self._rnn_hidden_size))

        self._rnn = torch.nn.GRU(
            input_size=self._rnn_hidden_size,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._num_layers)

        self._mlp2 = torch.nn.Sequential(
            torch.nn.Linear(self._rnn_hidden_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, output_size))

        self.reset_parameters()
        self.to(device)

    def reset_parameters(self):
        # initialization as described in: 
        # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                # use orthogonal initialization for weights
                torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                # set biases to zero
                torch.nn.init.zeros_(layer.bias)
            elif isinstance(layer, torch.nn.GRU):
                for name, param in layer.named_parameters():
                    if "weight" in name:
                        torch.nn.init.orthogonal_(param, 1.0)
                    elif "bias" in name:
                        torch.nn.init.zeros_(param)

    def forward(self, input, hidden):

        # include sequence dimension if not present
        added_seq_dim = False
        if input.ndim == 2:
            input = input[None, ...]
            added_seq_dim = True

        output = self._mlp1(input)
        output, hidden = self._rnn(output, hidden)
        output = self._mlp2(output)

        # remove sequence length dimension only if it was not present
        if added_seq_dim:
            output = torch.squeeze(output, dim=0)

        return output, hidden

    def init_hidden(self, batch_size):
        shape = ((self._num_layers, batch_size, self._rnn_hidden_size))
        return torch.zeros(shape, device=self._device)


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, (1, ))
    
    def action(self, act):
        return np.where(act <= 0, 0, 1)[0]


def eval(eval_env, agent, render=False):
    eval_hidden = agent.init_hidden(1)
    eval_obs = eval_env.reset()
    eval_obs = torch.from_numpy(eval_obs[..., :observation_space]).to(device)

    # evaluate performance
    eval_reward = 0.
    with torch.inference_mode():
        while True:
            action, eval_hidden = agent.get_action_inference(eval_obs.unsqueeze(0).unsqueeze(0), eval_hidden)
            eval_obs, reward, done, info = eval_env.step(action[0, 0].cpu().numpy())
            if render:
                eval_env.render()

            # convert to tensor, send to device and select subset of observation space
            eval_obs = torch.from_numpy(eval_obs[..., :observation_space]).to(device)
            eval_reward += reward

            if done:
                break

    print("Eval reward ", eval_reward)
    eval_env.close()
    return eval_reward


if __name__ == "__main__":

    # arguments
    seed = 5555
    num_updates = 50
    num_envs = 200
    num_steps = 500
    num_traj_minibatch = 200
    device = "cuda"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    envs = gym.vector.make('CartPole-v1', num_envs=num_envs, wrappers=DiscreteActions)
    envs.seed(seed)
    # can be less than full observation state of the environment
    # to make rnn useful
    observation_space = envs.single_observation_space.shape[0]
    action_space = envs.single_action_space.shape[0]

    # evaluation environment
    eval_env = gym.make('CartPole-v1')
    eval_env = DiscreteActions(eval_env)
    eval_env.seed(seed)

    # networks
    actor_model = RNN(
        input_size=observation_space,
        output_size=action_space,
        device=device)

    critic_model = RNN(
        input_size=observation_space,
        output_size=1,
        device=device)

    agent = ActorCritic(
        actor=actor_model, 
        critic=critic_model,
        recurrent_actor=True,
        recurrent_critic=True,
        actor_output_size=action_space,
        device=device)

    # PPO
    ppo = PPO(
        actor_critic=agent,
        recurrent=True,
        num_steps=num_steps,
        num_envs=num_envs,
        obs_size_actor=observation_space,
        obs_size_critic=observation_space,
        action_size=action_space,
        learning_rate=3e-4,
        num_traj_minibatch=num_traj_minibatch,
        update_epochs=4,
        use_gae=True,
        gae_lambda=0.95,
        gamma=0.99,
        clip_vloss=True,
        clip_coef=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        use_norm_adv=True,
        max_grad_norm=0.5,
        target_kl=0.01,
        learning_rate_gamma=0.999,
        min_seq_size=2,
        device=device)

    global_step = 0
    eval_rewards = []
    next_obs = torch.tensor(envs.reset()).to(device)
    next_dones = torch.zeros(num_envs).to(device)
    actor_state = agent.init_hidden(num_envs)
    critic_state = agent.init_hidden(num_envs)

    for update in range(1, num_updates + 1):
        print("Iter ", update)
        initial_actor_state = actor_state.clone()
        initial_critic_state = critic_state.clone()

        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs = next_obs
            dones = next_dones

            actions, logprobs, values, actor_state, critic_state = ppo.act(obs, obs, actor_state, critic_state)
            next_obs, rewards, next_dones, info = envs.step(actions.cpu().numpy())

            next_obs = torch.tensor(next_obs).to(device)
            rewards = torch.tensor(rewards).to(device)
            next_dones = torch.tensor(next_dones).to(device)

            # reset hidden states of dones
            actor_state[..., next_dones > 0.5, :] = 0.
            critic_state[..., next_dones > 0.5, :] = 0.

            ppo.set_step(obs, obs, actions, logprobs, rewards, dones, values)

        ppo.compute_returns(next_obs, next_dones, critic_state)

        loss, v_loss, pg_loss, entropy_loss, *_ = ppo.update(initial_actor_state, initial_critic_state)

        agent.clamp_std(0.2, 3.0)

        print("Total steps ", global_step)
        print("Total loss ", loss)
        print("Value loss ", v_loss)
        print("Policy loss ", pg_loss)
        print("Entropy loss ", entropy_loss)

        # evaluate
        if update % 1 == 0:
            eval_rewards.append(eval(eval_env, agent, render=False))

    envs.close()
