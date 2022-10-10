
import random

import gym
import numpy as np
import torch

from ppo import PPO, ActorCritic


class MLP(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            output_size: int,
            device: str):
        super().__init__()

        activation = torch.nn.LeakyReLU
        hidden_size = 64

        self._model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            activation(),
            torch.nn.Linear(hidden_size, hidden_size),
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
            if not isinstance(layer, torch.nn.Linear):
                continue
            # use orthogonal initialization for weights
            torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            # set biases to zero
            torch.nn.init.zeros_(layer.bias)

    def forward(self, input):
        output = self._model(input)
        return output


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, (1, ))
    
    def action(self, act):
        return np.where(act <= 0, 0, 1)[0]


def eval(eval_env, agent, render=False):
    eval_obs = eval_env.reset()
    eval_obs = torch.from_numpy(eval_obs[..., :observation_space]).to(device)

    # evaluate performance
    eval_reward = 0.
    with torch.inference_mode():
        while True:
            action, _ = agent.get_action_inference(eval_obs.unsqueeze(0).unsqueeze(0), None)
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
    actor_model = MLP(
        input_size=observation_space,
        output_size=action_space,
        device=device)

    critic_model = MLP(
        input_size=observation_space,
        output_size=1,
        device=device)

    agent = ActorCritic(
        actor=actor_model, 
        critic=critic_model,
        recurrent_actor=False,
        recurrent_critic=False,
        actor_output_size=action_space,
        device=device)

    # PPO
    ppo = PPO(
        actor_critic=agent,
        recurrent=False,
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
        min_seq_size=None,
        device=device)

    global_step = 0
    eval_rewards = []
    next_obs = torch.tensor(envs.reset()).to(device)
    next_dones = torch.zeros(num_envs).to(device)

    for update in range(1, num_updates + 1):
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs = next_obs
            dones = next_dones

            actions, logprobs, values, _, _ = ppo.act(obs, obs)
            next_obs, rewards, next_dones, info = envs.step(actions.cpu().numpy())

            next_obs = torch.tensor(next_obs).to(device)
            rewards = torch.tensor(rewards).to(device)
            next_dones = torch.tensor(next_dones).to(device)

            ppo.set_step(obs, obs, actions, logprobs, rewards, dones, values)

        ppo.compute_returns(next_obs, next_dones)

        loss, v_loss, pg_loss, entropy_loss = ppo.update()

        agent.clamp_std(0.2, 3.0)

        print("Iter ", update)
        print("Total steps ", global_step)
        print("Total loss ", loss)
        print("Value loss ", v_loss)
        print("Policy loss ", pg_loss)
        print("Entropy loss ", entropy_loss)

        # evaluate
        if update % 10 == 0:
            eval_rewards.append(eval(eval_env, agent, render=False))

    envs.close()
