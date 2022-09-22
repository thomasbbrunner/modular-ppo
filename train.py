# implementation of PPO based on:
# https://github.com/vwxyzjn/cleanrl
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

import argparse
import copy
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ppo import PPO, ActorCritic

from uncertainty_networks.uncertainty_networks import UncertaintyNetwork


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, (1, ))
    
    def action(self, act):
        return np.where(act <= 0, 0, 1)[0]


def eval(eval_env, policy, render=False):

    policy.eval()
    eval_hidden = policy.init_hidden(1)

    eval_obs = eval_env.reset()
    eval_obs = torch.from_numpy(eval_obs[..., :observation_space]).to(device)

    # evaluate performance
    eval_reward = 0.
    with torch.inference_mode():
        while True:
            action, eval_hidden = policy(eval_obs.unsqueeze(0).unsqueeze(0), eval_hidden)
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
    num_envs = 50
    num_steps = 500
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
    rnn_hidden_size = 5
    input_size = [observation_space, 5, rnn_hidden_size]
    hidden_size = [[64, 64], rnn_hidden_size, [64, 64]]
    output_size = [5, action_space]
    num_layers = 2
    dropout_prob = 0
    num_passes = 1
    num_models = 1

    actor_model = UncertaintyNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        num_passes=num_passes,
        num_models=num_models,
        initialization="rl",
        device=device)

    critic_model = UncertaintyNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=[output_size[0], 1],
        num_layers=num_layers,
        dropout_prob=dropout_prob,
        num_passes=num_passes,
        num_models=num_models,
        initialization="rl",
        device=device)

    agent = ActorCritic(
        actor=actor_model, 
        critic=critic_model,
        recurrent_critic=True,
        actor_output_size=action_space,
        min_seq_size=2,
        device=device)

    # PPO
    ppo = PPO(
        actor_critic=agent,
        num_steps=num_steps,
        num_envs=num_envs,
        obs_size_actor=observation_space,
        obs_size_critic=observation_space,
        action_size=action_space,
        learning_rate=3e-4,
        num_minibatches=2,
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
        device=device)

    global_step = 0
    eval_rewards = []
    next_obs = torch.tensor(envs.reset()).to(device)
    next_dones = torch.zeros(num_envs).to(device)
    actor_state = agent.init_hidden(num_envs)
    critic_state = agent.init_hidden(num_envs)

    for update in range(1, num_updates + 1):
        initial_actor_state = actor_state.clone()
        initial_critic_state = critic_state.clone()
        # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (update - 1.0) / num_updates
        #     lrnow = frac * args.learning_rate
        #     optimizer.param_groups[0]["lr"] = lrnow

        # TODO update learning rate
        for step in range(0, num_steps):
            global_step += 1 * num_envs
            obs = next_obs
            dones = next_dones

            actions, logprobs, values, actor_state, critic_state = ppo.act(obs, obs, actor_state, critic_state, dones)
            next_obs, rewards, next_dones, info = envs.step(actions.cpu().numpy())

            next_obs = torch.tensor(next_obs).to(device)
            rewards = torch.tensor(rewards).to(device)
            next_dones = torch.tensor(next_dones).to(device)

            ppo.set_step(obs, obs, actions, logprobs, rewards, dones, values)

        ppo.compute_returns(next_obs, critic_state, next_dones)

        v_loss, pg_loss, entropy_loss = ppo.update(initial_actor_state, initial_critic_state)

        agent.clamp_std(0.2, 3.0)

        print("Iter ", update)
        print("Total steps ", global_step)
        print("Value loss ", v_loss.item())
        print("Policy loss ", pg_loss.item())
        print("Entropy loss ", entropy_loss.item())

        # evaluate
        if update % 10 == 0:
            eval_rewards.append(eval(eval_env, agent.policy, render=False))

    envs.close()
