---

SimRealNet - Transforming Robotics
Abstract

---

Transferring reinforcement learning (RL) policies from simulation to the real world remains a significant challenge due to the reality gap in dynamics and observation noise. We propose SimRealNet, a novel domain-switching framework that adaptively alternates between simulation and real-world training based on a composite confidence score. This score integrates policy entropy, Q-ensemble variance, and Monte Carlo dropout uncertainty to robustly estimate policy reliability. To ensure smooth domain transitions and minimize oscillatory switching, SimRealNet employs an ARIMA-based temporal smoothing mechanism. Experiments across simulated and real robotic environments demonstrate that SimRealNet significantly improves sample efficiency, policy robustness, and training stability compared to standard hybrid RL baselines.

---

1. Introduction
The success of reinforcement learning (RL) in simulation often fails to translate directly to real-world settings due to domain shift and model discrepancies - commonly referred to as the reality gap. Simulators offer inexpensive and safe data generation, but cannot capture every nuance of real-world physics.
Key Question: How can an RL agent dynamically decide when to train in simulation versus in the real environment to optimize both learning speed and final performance?
In this work, we introduce SimRealNet, a principled domain-switching module that dynamically evaluates confidence in policy execution and uses it to guide when to engage the real robot. This design achieves three goals:
Minimize unnecessary real-world interactions
Maximize safe policy improvement
Stabilize training under domain shifts

---

2. Related Work
Sim2Real Transfer: Prior works such as Domain Randomization [Tobin et al., 2017] and System Identification [Christiano et al., 2016] attempt to bridge the gap by making simulation more diverse or realistic.
Uncertainty Estimation: Techniques like policy entropy [Haarnoja et al., 2018] and ensemble methods [Lakshminarayanan et al., 2017] estimate policy confidence, guiding cautious decision-making.
Hybrid Training: Recent approaches [Zhao et al., 2021] explore hybridization of simulated and real-world data without explicit adaptive switching.

Unlike previous methods, SimRealNet combines multiple uncertainty metrics and introduces a temporal smoothing process using ARIMA forecasting to regulate domain switches.

---

3. SimRealNet Framework
3.1 Confidence Estimation Module
SimRealNet evaluates three confidence scores:
Policy Entropy Confidence
 Measures the determinism of action selection:

where H(π(a∣s)) is the entropy over actions.MC Dropout Confidence
 Estimates predictive uncertainty via multiple stochastic forward passes:

Q-Ensemble Confidence
 Evaluates epistemic uncertainty by variance across NNN Q-networks:

Each score is normalized and combined using adaptive softmax weighting over uncertainty magnitudes:
3.2 Temporal Smoothing: ARIMA Forecasting
Raw confidence scores are noisy. SimRealNet applies an ARIMA(1,1,1) model fitted over recent reward history to predict upcoming reward trends and prevent premature domain switching.
The ARIMA-predicted trend influences switching via a dynamic adjustment factor β.
3.3 Domain Switching Logic
Enter Real Domain: If composite confidence exceeds a high threshold CenterC_{\text{enter}}Center​.
Exit Real Domain: If composite confidence falls below a low threshold CexitC_{\text{exit}}Cexit​ or if ARIMA reward forecast indicates impending degradation.
Hysteresis prevents oscillatory transitions by requiring distinct thresholds.

Pseudocode
If (Confidence > C_enter) and (not in_real):
    Switch to Real
Elif (Confidence < C_exit) or (ARIMA_Forecast < Beta):
    Switch to Sim

---

4. Experiments
4.1 Setup
Environments:
Simulation: Mujoco-based quadruped locomotion (custom domain randomization)
Real: Mini Cheetah robot on uneven terrain

Baseline Comparisons:
Sim-Only training
Real-Only training
Naïve Hybrid (fixed 1:1 sim-real switching)

Policies: SAC (Soft Actor Critic) with standard hyperparameters
Hardware: NVIDIA RTX 4090 for simulation, onboard Jetson Orin for real-world robot control.

---

4.2 Metrics
MetricDescriptionSuccess RateTask completion percentageReal Sample CountTotal real-world interaction stepsSwitching FrequencyNumber of sim-to-real transitionsReward VarianceTraining stability over episodes

---

4.3 Results
Key Findings:
SimRealNet achieves highest final performance with least real-world data.
Adaptive confidence avoids unnecessary risk in the real world.
ARIMA smoothing reduces domain jittering and switching frequency.

---

5. Ablation Studies

---

6. Conclusion
SimRealNet introduces a modular, generalizable domain-switching module for reinforcement learning under sim-to-real settings. By fusing composite uncertainty metrics with statistical temporal smoothing, it delivers safer, faster, and more stable training.
Here's the Github Repository : https://github.com/SaiNideeshKotagudem/SimRealNet
Please Visit and Share if possible.

---

7. Future Work
Scaling SimRealNet to high-dimensional vision-based policies.
Adapting meta-learned thresholds instead of manual Center, Cexit​.
Extending to multi-agent systems with coordinated domain switching.

---

Figures:
SimReaLNet Architecture OverviewARIMA Smoothing LogicImage depicting BaselineVsSimRealNet Rewards and Confidence and Reward vs Confidence Graph
