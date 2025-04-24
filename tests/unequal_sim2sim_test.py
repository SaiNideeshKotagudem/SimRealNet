import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pybullet_envs
import pybullet_envs.gym_locomotion_envs
import gym
import os
from SimRealNet import sim2real
import pyvirtualdisplay
# === SAC Components ===

class SACPolicyNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(action_dim, activation='tanh')  # Changed to tanh for continuous actions
        ])

    def call(self, state):
        return self.net(state)

class QNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1)  # Changed to 1 output for Q-value
        ])

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        return self.net(x)

class DropoutQNetwork(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(1)  # Changed to 1 output for Q-value

    def call(self, state, action, training=False):
        x = tf.concat([state, action], axis=-1)
        x = self.fc1(x)
        x = self.dropout(x, training=training)
        return self.out(x)

# === Visualization ===

def preprocess_obs(obs, target_dim):
    # Trims or pads the observation to match target_dim
    if obs.shape[0] > target_dim:
        return obs[:target_dim]
    elif obs.shape[0] < target_dim:
        padding = np.zeros(target_dim - obs.shape[0])
        return np.concatenate([obs, padding])
    return obs

def visualize_agent():
    _display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
    _ = _display.start()
    os.environ["DISPLAY"] = f":{_display.display}"
    
    real_env = gym.make("Walker2d-v4", render_mode='rgb_array')
    sim_env = gym.make("Walker2DBulletEnv-v0")

    state_dim = sim_env.observation_space.shape[0]  # 17
    action_dim = sim_env.action_space.shape[0]      # e.g., 6
    
    print("State dim:", state_dim, "Action dim:", action_dim)
    
    policy_net = SACPolicyNetwork(state_dim, action_dim)
    q_ensemble = [QNetwork(state_dim, action_dim) for _ in range(5)]
    dropout_qnet = DropoutQNetwork(state_dim, action_dim)
    evaluator = sim2real.ConfidenceEvaluator()

    # Pass the preprocess function into agent_loop if it accepts one
    confidence_history, env_situation, reward_history = sim2real.agent_loop(
        sim_env, real_env, policy_net, q_ensemble, dropout_qnet, evaluator,
        Enter_C=0.995, Exit_C=0.994, num_steps=100, ARIMA_beta=0.9,
        preprocess_fn=lambda obs: preprocess_obs(obs, target_dim=17)
    )

    x = range(len(confidence_history))
    color_flag = env_situation
    y1 = confidence_history
    y2 = reward_history
    colors = ['red' if flag else 'green' for flag in color_flag]

    plt.figure(figsize=(10, 6))
    for i in range(len(y1)):
        plt.scatter(i + 1, y1[i], color=colors[i], label='Confidence History' if i == 0 else '')
    plt.plot(range(1, len(y1) + 1), y1, color='blue', linewidth=1)

    for i in range(len(y2)):
        plt.scatter(i + 1, y2[i], color=colors[i], label='Reward History' if i == 0 else '', marker='s')
    plt.plot(range(1, len(y2) + 1), y2, color='orange', linewidth=1)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Time Step")
    plt.ylabel("Values")
    plt.title("Confidence and Reward Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print("Red Says In Real-world")
    print("Green Says In Simulation")
visualize_agent()
