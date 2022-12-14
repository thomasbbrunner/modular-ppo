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

from uncertainty_networks.uncertainty_networks import UncertaintyNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=5555,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")

    # Algorithm specific arguments
    parser.add_argument("--num_updates", type=int, default=50,
        help="number of optimization steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=50,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=500,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=0.01,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


class RecurrentActorCritic(torch.nn.Module):
    """
    Recurrent Actor Critic Module
    """

    def __init__(
            self,
            actor,
            critic,
            output_size,
            device):

        super().__init__()

        self._actor = actor
        self._critic = critic
        self._output_size = output_size

        self._log_std = torch.nn.Parameter(torch.zeros(output_size, device="cuda"))

    def _process_sequence(self, model, obs, hidden_states, dones):

        # input must be flattened
        assert obs.ndim == 2

        batch_size = hidden_states.shape[-2]
        seq_len = obs.shape[0] // batch_size
        # reshape to (sequence, batch)
        obs = obs.reshape(seq_len, batch_size, *obs.shape[1:])
        dones = dones.reshape(seq_len, batch_size, *dones.shape[1:])
        # clone dones as we will modify values in this array
        # TODO needed?
        hidden_states = hidden_states.clone()

        # split and pad sequences
        if seq_len > 1:
            # this tensor will be modified
            split_indices = dones.clone()
            # force split at the start of trajectory
            split_indices[0] = 1
            # transpose and flatten tensor to reorder entries
            # required later for splitting the observations correctly
            # for observation with indices (sequence, batch):
            #   old shape: [obs(0,0), obs(0,1), ...]
            #   new shape: [obs(0,0), obs(1,0), ...]
            obs = obs.movedim(0, 1).flatten(0, 1)
            dones = dones.movedim(0, 1).flatten(0, 1)
            split_indices = split_indices.movedim(0, 1).flatten(0, 1)
            # get indices (split requires tensor in CPU)
            split_indices = torch.argwhere(split_indices).squeeze().cpu()
            # split trajectories at dones or start of trajectory
            obs_split = torch.tensor_split(obs, split_indices, dim=0)
            # remove first element if empty
            if obs_split[0].numel() == 0:
                obs_split = obs_split[1:]
            # pad and join trajectories
            obs = torch.nn.utils.rnn.pad_sequence(obs_split)

            # add hidden states to newly created batches
            h = self.init_hidden(len(obs_split))
            # get indices of started trajectories
            started_traj_indices = seq_len*torch.arange(batch_size)
            # remove started trajectories that have a done
            started_traj_mask = ~(dones[started_traj_indices] > 0.5)
            started_traj_indices = started_traj_indices[started_traj_mask]
            # batches corresponding to started trajectories
            started_traj_batches = torch.argwhere(torch.isin(split_indices, started_traj_indices)).flatten()
            # insert existing hidden states of started trajectories
            h[..., started_traj_batches, :] = hidden_states[..., started_traj_mask, :]
            hidden_states = h

        # reset hidden states of done envs
        else:
            hidden_states[..., (dones > 0.5).squeeze(0), :] = 0.

        output, hidden_states = model(obs, hidden_states)

        if seq_len > 1:
            # get start and end of each trajectory
            # for this we need to add ending of last trajectory to split_indices
            split_indices = torch.cat((split_indices, torch.tensor([batch_size*seq_len])))
            # get original length of each trajectory
            traj_lengths = torch.diff(split_indices)
            # get mask of valid trajectories
            traj_mask = traj_lengths > torch.arange(0, seq_len).unsqueeze(1)
            # truncate mask
            traj_mask = traj_mask[:output.shape[0]]
            # apply mask
            output = output.movedim(0, 1)[traj_mask.movedim(0, 1)].reshape(batch_size, seq_len, -1).movedim(0, 1).flatten(0, 1)

            # in this case, hidden sizes make no sense, as we're iterating over padded obs
            hidden_states = None

        else:
            # flattten sequence and batch dimensions
            output = output.flatten(0, 1)

        assert output.shape[0] == seq_len*batch_size

        return output, hidden_states

    def _iterate_sequence(self, model, obs, hidden_states, dones):

        # unflatten the first dimension
        batch_size = hidden_states.shape[-2]
        obs = obs.reshape(-1, batch_size, *obs.shape[1:])
        dones = dones.reshape(-1, batch_size, *dones.shape[1:])
        outputs = []

        for o, d in zip(obs, dones):
            # reset hidden states of done envs
            hidden_states[..., d.bool(), :] *= 0.
            # add sequence dimension and pass to model
            output, hidden_states = model(o.unsqueeze(0), hidden_states)
            outputs.append(output)

        # flattten sequence and batch dimensions
        outputs = torch.flatten(torch.cat(outputs), 0, 1)
        return outputs, hidden_states

    def get_value(self, obs, hidden_states, dones):
        value, hidden_states = self._process_sequence(self._critic, obs, hidden_states, dones)
        return value, hidden_states

    def get_action(self, obs, hidden_states, dones, action=None):
        action_mean, hidden_states = self._process_sequence(self._actor, obs.clone(), hidden_states.clone(), dones.clone())
        probs = torch.distributions.Normal(action_mean, torch.exp(self._log_std))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), hidden_states

    def init_hidden(self, batch_size=None):
        return self._actor.init_hidden(batch_size)

    @property
    def policy(self):
        return copy.deepcopy(self._actor)


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

    args = parse_args()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    envs = gym.vector.make('CartPole-v1', num_envs=args.num_envs, wrappers=DiscreteActions)
    envs.seed(args.seed)
    # can be less than full observation state of the environment
    # to make rnn useful
    observation_space = envs.single_observation_space.shape[0]
    action_space = envs.single_action_space.shape[0]

    # evaluation environment
    eval_env = gym.make('CartPole-v1')
    eval_env = DiscreteActions(eval_env)
    eval_env.seed(args.seed)

    # networks
    rnn_hidden_size = 5
    input_size = [observation_space, 5, rnn_hidden_size]
    hidden_size = [[64, 64], rnn_hidden_size, [64, 64]]
    output_size = [5, action_space]
    num_layers = 2
    dropout_prob = 0
    num_passes = 1
    num_models = 1
    num_steps = 100
    device = "cuda"

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

    agent = RecurrentActorCritic(
        actor=actor_model, 
        critic=critic_model,
        output_size=action_space,
        device=device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, observation_space)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_space)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    eval_rewards = []
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_state_actor = agent.init_hidden(args.num_envs)
    next_state_critic = agent.init_hidden(args.num_envs)
    num_updates = args.num_updates

    for update in range(1, num_updates + 1):
        initial_state_actor = next_state_actor.clone()
        initial_state_critic = next_state_critic.clone()
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, next_state_actor = agent.get_action(next_obs, next_state_actor, next_done)
                value, next_state_critic = agent.get_value(next_obs, next_state_critic, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value, _ = agent.get_value(
                next_obs,
                next_state_critic,
                next_done,
            )
            next_value = next_value.reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1, observation_space))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, action_space))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, _ = agent.get_action(
                    b_obs[mb_inds],
                    initial_state_actor[..., mbenvinds, :],
                    b_dones[mb_inds],
                    b_actions[mb_inds],
                )
                newvalue, _ = agent.get_value(
                    b_obs[mb_inds],
                    initial_state_critic[..., mbenvinds, :],
                    b_dones[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print("Iter ", update)
        print("Total steps ", global_step)
        print("losses/value_loss ", v_loss.item())
        print("losses/policy_loss ", pg_loss.item())

        # evaluate
        if update % 10 == 0:
            eval_rewards.append(eval(eval_env, agent.policy, render=False))

    envs.close()
