import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pybullet_envs
import pybullet_envs.gym_locomotion_envs
import gym
import os
import pyvirtualdisplay
from SimRealNet import sim2real

# === SAC Components ===

class SACPolicyNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='tanh')
        ])

    def call(self, state):
        return self.net(state)

class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        return self.net(x)

class DropoutQNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(1)

    def call(self, state, action, training=False):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.out(x)

# === Utility ===

def preprocess_obs(obs, target_dim):
    if obs.shape[0] > target_dim:
        return obs[:target_dim]
    elif obs.shape[0] < target_dim:
        padding = np.zeros(target_dim - obs.shape[0])
        return np.concatenate([obs, padding])
    return obs

# === SimRealNet Visualization + Baseline Comparison ===

def visualize_agent():
    _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
    _ = _display.start()
    os.environ["DISPLAY"] = f":{_display.display}"

    real_env = gym.make("Walker2d-v4", render_mode='rgb_array')
    sim_env = gym.make("Walker2DBulletEnv-v0")

    state_dim = sim_env.observation_space.shape[0]
    action_dim = sim_env.action_space.shape[0]

    print("State dim:", state_dim, "Action dim:", action_dim)

    policy_net = SACPolicyNetwork(state_dim, action_dim)
    q_ensemble = [QNetwork(state_dim, action_dim) for _ in range(5)]
    dropout_qnet = DropoutQNetwork(state_dim, action_dim)
    evaluator = sim2real.ConfidenceEvaluator()

    # SimRealNet run
    simreal_conf, simreal_env_flags, simreal_rewards = sim2real.agent_loop(
        sim_env, real_env, policy_net, q_ensemble, dropout_qnet, evaluator,
        Enter_C=0.995, Exit_C=0.994, num_steps=100, ARIMA_beta=0.9,
        preprocess_fn=lambda obs: preprocess_obs(obs, target_dim=17)
    )

    # Baseline run (pure simulation)
    baseline_rewards = []
    obs = sim_env.reset()
    for _ in range(100):
        obs = preprocess_obs(obs, 17)
        obs_tf = tf.convert_to_tensor(obs[None, :], dtype=tf.float32)
        action = policy_net(obs_tf)[0].numpy()
        obs, reward, done, _, _ = sim_env.step(action)
        baseline_rewards.append(reward)
        if done:
            obs = sim_env.reset()

    # === Plotting ===
    x = range(1, len(simreal_rewards) + 1)
    colors = ['red' if flag else 'green' for flag in simreal_env_flags]

    plt.figure(figsize=(12, 6))

    # Confidence Plot
    plt.subplot(2, 1, 1)
    for i, c in enumerate(simreal_conf):
        plt.scatter(i + 1, c, color=colors[i], label='Confidence' if i == 0 else '')
    plt.plot(x, simreal_conf, color='blue', label='Confidence Line')
    plt.ylabel("Confidence Score")
    plt.title("SimRealNet: Confidence and Reward over Time")
    plt.legend()
    plt.grid(True)

    # Reward Comparison Plot
    plt.subplot(2, 1, 2)
    plt.plot(x, simreal_rewards, label='SimRealNet Reward', color='orange')
    plt.plot(x, baseline_rewards, label='Baseline (Pure Simulation)', color='gray', linestyle='--')
    plt.xlabel("Time Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("Red = Real Environment | Green = Simulation")

visualize_agent()
