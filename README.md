# Confidence-Based Sim2Real Environment Switcher

This repository contains a novel approach to domain adaptation for reinforcement learning, utilizing **confidence-based switching** between simulated and real environments. The method allows for more robust training and seamless transitions from simulation to real-world tasks, making it ideal for robotics applications and other real-world RL systems.

## Table of Contents

* [Introduction](#introduction)
* [Key Features](#key-features)
* [Usage](#usage)
* [Installation](#installation)
* [Configuration](#configuration)
* [Training Loop](#training-loop)
* [License](#license)
* [Contact](#contact)

## Introduction

In reinforcement learning, a major challenge is transferring policies learned in simulations to real-world environments (Sim2Real). The discrepancy between the simulated environment and the real world, known as the **reality gap**, often results in suboptimal performance. This project presents a **confidence-based switching** mechanism that enables the RL agent to switch between simulated and real-world environments based on the **confidence** of the agent's learned policy.

By utilizing several confidence metrics—**policy entropy**, **Q-ensemble variance**, and **MC dropout**—the agent can intelligently decide when to trust the simulated environment and when to transition to the real-world environment, leading to improved robustness and better real-world performance.

## Key Features

* **Confidence-Based Environment Switching**: Seamlessly switches between simulated and real environments based on confidence values.
* **Flexible Configuration**: Easily adjustable parameters to fine-tune the confidence thresholds for switching.
* **Adaptive Confidence Calculation**: Uses a combination of policy entropy, Q-ensemble variance, and MC dropout to calculate a reliable confidence value.
* **Integrated with RL Frameworks**: Designed to integrate with existing reinforcement learning systems, particularly Soft Actor-Critic (SAC).
* **Support for any Simulated Environment**: Consists of a Modular approach.

## Usage

To use this repository in your project, follow these steps:

### 1. Import the Confidence Evaluator and Agent Loop:

```python
from confidence_based_sim2real import ConfidenceEvaluator, agent_loop
```

### 2. Initialize the environments:

```python
# Initialize simulated and real environments
sim_env = <YourSimulationEnv>()
real_env = <YourRealWorldEnv>()
```

### 3. Initialize the neural networks and other components:

```python
policy_net = <YourPolicyNetwork>()
q_ensemble = <YourQEnsembleNetwork>()
dropout_q_net = <YourMCDropoutNetwork>()

evaluator = ConfidenceEvaluator(sigma_max=1.0, adaptive=True)
```

### 4. Define confidence thresholds for switching:

```python
Enter_C = 0.8  # Confidence threshold for entering real environment
Exit_C = 0.3   # Confidence threshold for exiting real environment
```

### 5. Run the agent loop:

```python
agent_loop(sim_env, real_env, policy_net, q_ensemble, dropout_q_net, evaluator, Enter_C, Exit_C)
```

This will allow the agent to intelligently switch between the simulated and real environments based on its confidence in the action selection.

## Installation

To install the necessary dependencies for this project, you can use the following command:

```bash
pip install -r requirements.txt
```

Ensure you have the following dependencies:

* TensorFlow
* NumPy
* OpenAI Gym (for environment simulation)
* PyBullet (for simulation)
* MuJoCo (optional, for higher-fidelity simulation)

If you're using a specific RL framework like SAC, PPO, or DQN, you'll need to install the respective libraries.

## Configuration

You can modify the behavior of the environment switcher by adjusting these parameters:

* **sigma\_max**: Controls the maximum variance for Q-values during confidence evaluation.
* **adaptive**: Whether to use adaptive weights for confidence calculation. Set to `True` to automatically adjust weights based on the uncertainty in the system.

## Training Loop

This repository includes a training loop that integrates confidence-based environment switching. Here's how it works:

1. **Simulated Environment**: The agent is trained in the simulated environment and learns a policy.
2. **Confidence Calculation**: At each step, the confidence of the agent's action is evaluated using the three components: policy entropy, Q-ensemble variance, and MC dropout.
3. **Environment Switching**: Based on the confidence value, the agent either stays in the simulated environment or transitions to the real environment.
4. **Hysteresis Logic**: Once the agent transitions to the real environment, it requires a drop in confidence below a certain threshold to transition back to simulation.

The system adapts dynamically, ensuring that the agent is always in the optimal environment for learning.

## Further Information

1. Medium Blog - https://medium.com/@sainideesh.k/simrealnet-transforming-robotics-8efba1df6671
2. Wikipedia - Pending For Publication

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or would like to discuss this project further, feel free to reach out to me.

* Email: [sainideesh.k@gmail.com](mailto:sainideesh.k@gmail.com)
* GitHub: [SaiNideeshKotagudem](https://github.com/SaiNideeshKotagudem)


