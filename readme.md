# Modular PPO
PPO implementation based on [CleanRL](https://github.com/vwxyzjn/cleanrl).

Features:
- Support for recurrent policies
- Support for different actor and critic architectures
- Faster inference and training of recurrent policies by processing entire trajectories in a single call to PyTorch
- (optional) Training using mini-batches
- (optional) Truncated BPTT
- (optional) Discarding of short trajectories

## Installation
To install, run:
```
pip install .
```

## Usage
Examples for usage can be found in the `scripts` folder.
