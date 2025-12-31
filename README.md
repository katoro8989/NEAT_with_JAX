# NEAT with JAX

A collection of NEAT (NeuroEvolution of Augmenting Topologies) implementations using JAX, featuring experiments in both reinforcement learning and supervised learning domains.

## Overview

This repository contains two experimental scripts:

1. **SlimeVolley NEAT** (`slimevolley.ipynb`)
   Evolving agents to play SlimeVolleyGym against the internal AI using reinforcement learning

2. **Hybrid NEAT for 2D Classification** (`2D_claddification.ipynb`)
   A novel hybrid approach combining NEAT's structural evolution with gradient descent for 2D classification tasks

## Features

- **Fully GPU-Parallelized**: High-speed evolutionary computation leveraging JAX's `jit` and `vmap`
- **Speciation**: Niching strategy based on structural similarity
- **Dynamic Network Topology**: Incremental evolution through node and connection addition
- **Visualization**: Network structure and performance visualization

## Requirements

```bash
pip install jax jaxlib
pip install evojax  # For SlimeVolley
pip install numpy matplotlib networkx
```

## 1. SlimeVolley NEAT (`slimevolley.ipynb`)

### Overview
Evolves agents to compete against the internal AI in the SlimeVolleyGym environment.

### Key Parameters
- **Population Size**: 4096 individuals
- **Max Nodes**: 50
- **Input Dimensions**: 12 (observation space)
- **Output Dimensions**: 3 (action space)
- **Max Steps**: 3000 steps per episode

### Fitness Composition
The reward function consists of the following components:

- **Base Reward**: Game win/loss score
- **Ball Proximity Bonus**: Encourages approaching the ball
- **Forward Bonus**: Encourages agent forward movement
- **Opponent Court Bonus**: Rewards returning the ball to opponent's court
- **Survival Bonus**: Encourages longer episode survival

### Usage
```python
population, fitness_np_pure, max_pure_hist, mean_pure_hist = main()
```

### Output
- Network structure visualization of the best individual every 10 generations (`best_genome_gen_*.png`)
- Fitness scores per generation (both with bonuses and pure scores)

## 2. Hybrid NEAT for 2D Classification (`2D_claddification.ipynb`)

### Overview
Solves 2D classification problems using a novel hybrid approach combining NEAT's structure search with gradient descent.

### Key Parameters
- **Population Size**: 1024 individuals
- **Max Nodes**: 32
- **Input Dimensions**: 2 (x, y coordinates)
- **Output Dimensions**: 1 (classification probability)
- **Training Epochs**: 20 epochs per individual per generation

### Supported Datasets
- **XOR**: Classic benchmark for non-linear separability
- **Circle**: Classification with circular boundary
- **Spiral**: Spiral pattern classification

### Hybrid Approach
1. **Evolution Phase**: NEAT explores network topology
   - Node addition, link addition, weight mutation
   - Diversity maintenance through speciation

2. **Learning Phase**: Gradient descent optimizes weights for each individual
   - Binary cross-entropy loss
   - SGD-style updates (learning rate: 0.05)

### Fitness Function
```
fitness = accuracy - (PENALTY_NODE × num_nodes + PENALTY_LINK × num_links)
```
Maximizes accuracy while penalizing network complexity.

### Usage
```python
# Select dataset ("xor", "circle", "spiral")
DATASET_NAME = "spiral"
DATA_X, DATA_Y = get_dataset(DATASET_NAME, BATCH_SIZE)

# Run
main()
```

### Output
- Decision boundary and network structure visualization (every 10 generations)
- Maximum/mean accuracy and fitness per generation