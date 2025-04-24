import numpy as np
import tensorflow as tf
from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA

# ===== Confidence Evaluator =====
class sim2real:

    class ConfidenceEvaluator:
        def __init__(self, sigma_max=1.0, adaptive=True):
            self.sigma_max = sigma_max
            self.adaptive = adaptive

        def policy_entropy_confidence(self, action_probs):
            entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-8))
            max_entropy = tf.math.log(tf.cast(tf.shape(action_probs)[0], tf.float32))
            return 1.0 - entropy / max_entropy

        def q_ensemble_confidence(self, q_values_ensemble):
            var = tf.math.reduce_variance(q_values_ensemble)
            return 1.0 - var / self.sigma_max

        def mc_dropout_confidence(self, q_values_mc):
            var = tf.math.reduce_variance(q_values_mc)
            return 1.0 - var / self.sigma_max

        def adaptive_weights(self, uncertainties):
            uncertainties = tf.stack(uncertainties)
            uncertainties = uncertainties / (tf.reduce_max(uncertainties) + 1e-8)
            return tf.nn.softmax(1.0 / (uncertainties + 1e-8))

        def compute(self, c1, c2, c3):
            if self.adaptive:
                raw_uncertainty = [1.0 - c1, 1.0 - c2, 1.0 - c3]
                weights = self.adaptive_weights(raw_uncertainty)
                return weights[0]*c1 + weights[1]*c2 + weights[2]*c3
            else:
                return (c1 + c2 + c3) / 3.0


#=====ARIMA Smoothing Logic======#
    def ARIMA(data):
        if len(data) > 2:
            model = StatsmodelsARIMA(data, order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return forecast[0]
        else:
            return float('inf')


# === Agent Loop with Switching & Hysteresis ===
    def agent_loop(sim_env, real_env, policy_net, q_ensemble, dropout_q_net, evaluator,
               Enter_C, Exit_C, num_steps, ARIMA_beta, preprocess_fn=None):
        CONFIDENCE_ENTER_REAL = Enter_C
        CONFIDENCE_EXIT_REAL = Exit_C

        env_situation = []
        confidence_history = []
        reward_history = []

        in_real = False
        state = sim_env.reset()
        if preprocess_fn:
            state = preprocess_fn(state)

        for step in range(num_steps):
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), tf.float32)
            action = policy_net(state_tensor)[0].numpy()

        # For entropy calculation (approximate for continuous actions)
            action_probs = tf.nn.softmax(action) if len(action.shape) == 1 else action

        # Confidence components
            c1 = evaluator.policy_entropy_confidence(action_probs)
            q_vals = tf.stack([q_net(state_tensor, tf.expand_dims(action, 0))[0] for q_net in q_ensemble])
            c2 = evaluator.q_ensemble_confidence(q_vals)
            mc_vals = tf.stack([dropout_q_net(state_tensor, tf.expand_dims(action, 0), training=True)[0] for _ in range(10)])
            c3 = evaluator.mc_dropout_confidence(mc_vals)

            conf = evaluator.compute(c1, c2, c3)
            confidence_history.append(conf.numpy())

            if in_real and conf < CONFIDENCE_EXIT_REAL:
                in_real = False
            elif not in_real and conf > CONFIDENCE_ENTER_REAL:
                in_real = True
            elif in_real and c_value.ARIMA(reward_history) <= ARIMA_beta:
                in_real = False

        # Reset real env only on transition to real
            if in_real and (not env_situation or not env_situation[-1]):
                state = real_env.reset()
                if preprocess_fn:
                    state = preprocess_fn(state)

            env_situation.append(in_real)
            env = real_env if in_real else sim_env

            next_state, reward, done, _ = env.step(action)
            if preprocess_fn:
                next_state = preprocess_fn(next_state)

            reward_history.append(np.clip(reward, -1, 1))
            state = next_state

            if done:
                state = env.reset()
                if preprocess_fn:
                    state = preprocess_fn(state)

        return confidence_history, env_situation, reward_history
