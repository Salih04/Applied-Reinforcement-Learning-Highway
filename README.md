# Applied Reinforcement Learning
## Autonomous Driving with highway-env (DQN)

**Author**
- Salih Camcı | 2202041

> **Track:** Autonomous Driving (highway-env)  
> **Environment:** `highway-fast-v0` 

---

## Evolution Video (3 Stages)

Same env config + same seed, three checkpoints:

1) **Untrained** : Random actions, crashes quickly  
2) **Half-trained** : Survives longer, but still makes risky lane changes  
3) **Fully trained** : More stable high speed driving with fewer collisions

![Evolution](assets/evolution.gif)

---

## Objective

Train an agent that drives as fast as possible in **dense traffic** while avoiding crashes.  
The main challenge is balancing speed vs. safety.

---

## Environment Setup

- Environment: `highway-fast-v0`
- Lanes: 4
- Traffic density: high (approximately 30 vehicles)
- Episode duration: 40 steps
- CPU-friendly configuration

---

## Methodology

### Observations (State)

I use Kinematics observations.  
The agent observes a fixed-size list of nearby vehicles (kinematics features):

- presence
- relative position (x, y)
- relative velocity (vx, vy)

This is lightweight and trains fast on CPU.

### Actions

I use **DiscreteMetaAction**:

- `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`

This is an abstraction of a simple throttle/ steering and simplifies learning.

### Algorithm

I use DQN (Stable-Baselines3) because:
- the action space is discrete
- it runs efficiently on CPU
- it’s a common baseline for highway-env

**Network**
- MLP with `[256, 256]` hidden layers (ReLU)

---

## Reward Function (Custom Shaping)

The default rewards were effective, but the initial training was too irresponsible: the agent educated to run after speed and to crash very often.
To address this, a custom reward shaping function was designed to explicitly balance speed, safety, and driving smoothness.
The shaping does not introduce new information, but reweights existing signals to stabilize learning and reduce unsafe behaviors.

### Reward Function (Custom Shaping)

R_t = R_env_t + α·r_speed(t) + β·r_lane(t) − γ·I[collision] − δ·I[d_min(t) < d_unsafe] − λ·I[lane_change]

Plain-text version:
R_t = R_env_t
    + alpha * r_speed(t)
    + beta  * r_lane(t)
    - gamma * I[collision]
    - delta * I[d_min(t) < d_unsafe]
    - lambda* I[lane_change]

Where:
- r_speed(t) ∈ [0, 1]: normalized speed reward
- r_lane(t) ∈ [0, 1]: lane preference (stability / keep-lane)
- I[·]: indicator function (1 if condition is true, else 0)
- d_min(t): closest front-vehicle distance in the same lane


---

## Hyperparameters (Main Ones)
- total timesteps: **300,000**
- learning rate: **5e-4**
- buffer size: **50,000**
- batch size: **64**
- gamma: **0.99**
- target update: **1000 steps**
- exploration:
  - fraction: **0.2**
  - final epsilon: **0.05**

Why these values?
- I started close to SB3 examples, then adjusted mainly based on stability of the reward curve.
- Too aggressive exploration early on caused frequent crashes; the shaping + epsilon schedule helped.

---

## Training Analysis

![Reward Curve](assets/reward_curve.png)

**What the curve shows (brief):**
- Early phase: Low reward because the policy is basically random 
- Middle phase: Reward rises as the agent learns “don’t crash immediately”
- Late phase: Curve stabilizes; improvements become smaller (policy converges under current setup)

---

## Challenges & Fixes

### 1) Too many early crashes
At first, the agent often spammed `FASTER` and changed lanes aggressively, which caused lots of collisions. I fixed this by:
- increasing collision penalty
- adding an unsafe following penalty
- adding a lane change penalty

This not only enabled the behavior to become significantly smoother but also did not slow down the agent.

---

## How to Run 

### 1) Install
```python
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Train
This will train for 300k timesteps and save checkpoints under models/:
- models/dqn_half.zip
- models/dqn_full.zip
  
```python
python -m src.train
```
### 3) Evaluate 
```python
python -m src.evaluate --model models/dqn_full.zip
```

### 4) Plot reward curve
```python
python -m src.plot_rewards
```
---

### Generate evolution video
The evolution GIF was created from short screen recordings of three different agent checkpoints.
Raw video files are not included in the repository to keep it lightweight.

#### Untrained (random policy)
```python
python -m src.play --seed 0 --record
```
#### Half-trained
```python
python -m src.play --seed 0 --record --model models/dqn_half.zip
```
#### Fully trained
```python
python -m src.play --seed 0 --record --model models/dqn_full.zip
```
