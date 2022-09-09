
import copy
import numpy as np
import torch


class ActorCritic(torch.nn.Module):
    """
    """

    def __init__(
            self,
            actor: torch.nn.Module,
            critic: torch.nn.Module,
            actor_output_size: int,
            min_seq_size: int,
            device: str):

        super().__init__()

        self._actor = actor
        self._critic = critic
        self._actor_output_size = actor_output_size
        self._min_seq_size = min_seq_size

        self._log_std = torch.nn.Parameter(torch.zeros(actor_output_size, device=device))

    def _process_sequence(self, model, obs, hidden_states, dones):
        """
        Processes a batch of sequences in a computationally efficient way by
        splitting a sequence containing several trajectories into several
        sequences that contain only one trajectory.
        This is accomplished by splitting the input sequence everywhere where
        there's a done and padding the rest of the sequence.

        Visualization:
        Original sequences: [[a1, a2, a3, a4 | a5, a6],
                             [b1, b2 | b3, b4, b5 | b6]]

        Split sequences:    [[a1, a2, a3, a4],
                             [a5, a6, 0, 0],
                             [b1, b2, 0, 0],
                             [b3, b4, b5, 0],
                             [b6, 0, 0, 0]]

        Attention: this method can lead to very large batch sizes. This becomes
        a problem if input contains many small trajectories. For this reason,
        trajectories shorter than `_min_seq_size` are removed.
        """

        # input must be flattened
        assert obs.ndim == 2

        batch_size = hidden_states.shape[-2]
        seq_len = obs.shape[0] // batch_size
        # reshape to (sequence, batch)
        obs = obs.reshape(seq_len, batch_size, *obs.shape[1:])
        dones = dones.reshape(seq_len, batch_size, *dones.shape[1:])

        if seq_len > 1:
            # SPLIT TRAJECTORIES
            # clone, as this tensor will be modified
            split_indices = dones.clone()
            # force split at the start of trajectory
            split_indices[0] = 1
            # transpose and flatten tensor to reorder entries
            # required later for splitting the observations correctly
            # for observation with indices (sequence, batch):
            #   old shape: [obs(0,0), obs(0,1), ...]
            #   new shape: [obs(0,0), obs(1,0), ...]
            # causes data to be copied
            obs = obs.movedim(0, 1).flatten(0, 1)
            dones = dones.movedim(0, 1).flatten(0, 1)
            split_indices = split_indices.movedim(0, 1).flatten(0, 1)
            # get indices (split requires tensor in CPU)
            split_indices = torch.argwhere(split_indices).flatten().cpu()
            # split trajectories at dones or start of trajectory
            obs_split = torch.tensor_split(obs, split_indices, dim=0)
            # remove first element if empty
            if obs_split[0].numel() == 0:
                obs_split = obs_split[1:]

            # REMOVE SHORT TRAJECTORIES
            # get start and end of each trajectory
            # for this we need to add the end of last trajectory to split_indices
            split_indices = torch.cat((split_indices, torch.tensor([batch_size*seq_len])))
            # get original length of each trajectory
            traj_lengths = torch.diff(split_indices)
            # select trajectories to remove (smaller than x steps)
            keep_mask = traj_lengths >= self._min_seq_size
            keep_indices = torch.argwhere(keep_mask).flatten()
            # filter trajectories only if needed
            if not torch.all(keep_mask):
                if keep_indices.numel() == 0:
                    # TODO don't know what happens in this case
                    breakpoint()
                obs_split = [obs_split[i.item()] for i in keep_indices]

            # PAD AND JOIN TRAJECTORIES
            obs = torch.nn.utils.rnn.pad_sequence(obs_split)

            # GENERATE HIDDEN STATES
            # get indices of started trajectories
            # (the ones that have an initial hidden state)
            started_traj_indices = seq_len*torch.arange(batch_size)
            # batches corresponding to started trajectories
            started_traj_batches = torch.argwhere(torch.isin(split_indices, started_traj_indices)).flatten()
            # generate mask for started trajectories that have a done
            # (these should be discarded)
            started_done_traj_mask = torch.logical_not(dones[started_traj_indices] > 0.5).cpu()
            # generate mask for new hidden states only with relevant batches
            new_batches_mask = torch.isin(keep_indices, started_traj_batches[started_done_traj_mask])
            # generate mask for hidden states of started trajectories
            started_batches_mask = torch.isin(started_traj_batches, keep_indices)
            started_batches_mask = torch.logical_and(started_batches_mask, started_done_traj_mask)
            # insert existing hidden states of started trajectories
            h = self.init_hidden(len(obs_split))
            h[..., new_batches_mask, :] = hidden_states[..., started_batches_mask, :]
            hidden_states = h

            print("Num trajectories before/after prune: ", traj_lengths.shape[0], "/", len(obs_split))

        else:
            # reset hidden states of done envs
            hidden_states[..., (dones > 0.5).squeeze(0), :] = 0.

        output, hidden_states = model(obs, hidden_states)

        if seq_len > 1:
            # UNPAD AND JOIN OUTPUTS
            # get mask of valid trajectories
            traj_mask = traj_lengths > torch.arange(0, seq_len).unsqueeze(1)
            # create array with full size of output
            # (as if short trajectories were not removed)
            output_full = torch.zeros((*traj_mask.shape, output.shape[-1]), device=output.device)
            # move data
            output_full[:output.shape[0], keep_indices, :] = output
            output = output_full
            # apply mask to unpad outputs and join them
            output = output.movedim(0, 1)[traj_mask.movedim(0, 1)].reshape(batch_size, seq_len, -1).movedim(0, 1).flatten(0, 1)

            # in this case, hidden sizes make no sense, as we're iterating over padded obs
            hidden_states = None

        else:
            # flattten sequence and batch dimensions
            output = output.flatten(0, 1)

        assert output.shape[0] == seq_len*batch_size

        return output, hidden_states

    def _iterate_sequence(self, model, obs, hidden_states, dones):
        """
        Processes a batch of sequences by iterating over all steps in the
        sequence. This approach is slower than the one in `_process_sequence`,
        but has a smaller memory footprint.
        """
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
        action_mean, hidden_states = self._process_sequence(self._actor, obs, hidden_states, dones)
        probs = torch.distributions.Normal(action_mean, torch.exp(self._log_std))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), hidden_states

    def init_hidden(self, batch_size=None):
        return self._actor.init_hidden(batch_size)

    def clamp_std(self, min_std, max_std):
        self._log_std.detach().clamp_(np.log(min_std), np.log(max_std))

    @property
    def policy(self):
        return copy.deepcopy(self._actor)
