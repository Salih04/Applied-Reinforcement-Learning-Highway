# Applied Reinforcement Learning
## Autonomous Driving with highway-env (DQN)

**Group Members**
- Salih Camcı

> **Track:** Autonomous Driving (highway-env)  
> **Environment:** `highway-fast-v0` (CPU-friendly)

---

## Evolution Video (3 Stages)

Same env config + same seed, three checkpoints:

1) **Untrained** — mostly random actions, crashes quickly  
2) **Half-trained** — survives longer, but still makes risky lane changes  
3) **Fully trained** — more stable high-speed driving with fewer collisions

![Evolution](assets/evolution.gif)

---

## Objective

Train an agent that drives **as fast as possible** in **dense traffic** while **avoiding crashes**.  
The main challenge is balancing **speed vs. safety**.

---

## Methodology

### Observations (State)

We use **Kinematics** observations (vector-based, not pixels).  
The agent sees a fixed-size list of nearby vehicles with:

- presence
- relative position `(x, y)`
- relative velocity `(vx, vy)`
- heading `(cos(h), sin(h))`

This is lightweight and trains fast on CPU.

### Actions

We use **DiscreteMetaAction**:

- `LANE_LEFT`, `IDLE`, `LANE_RIGHT`, `FASTER`, `SLOWER`

This abstracts low-level throttle/steering and makes learning easier.

### Algorithm

We use **DQN (Stable-Baselines3)** because:
- the action space is discrete
- it runs efficiently on CPU
- it’s a common baseline for highway-env

**Network**
- MLP with `[256, 256]` hidden layers (ReLU)

---

## Reward Function (Custom Shaping)

The default rewards worked, but early training was too “reckless”: the agent learned to chase speed and crash frequently.
To address this, we added a simple reward shaping term that balances speed, safety, and smooth driving.

$$
R_t =
\alpha \cdot r_{\text{speed}}
+ \beta \cdot r_{\text{right}}
- \gamma \cdot \mathbb{1}[\text{collision}]
- \delta \cdot \mathbb{1}[d_{\min} < d_{\text{unsafe}}]
- \lambda \cdot \mathbb{1}[\text{lane\_change}]
$$


**Reward components:**
- $r_{\text{speed}}$: normalized speed reward (mapped from the speed range)
- $r_{\text{right}}$: right-lane preference reward
- $\mathbb{1}[\text{collision}]$: collision penalty (discourages crashing behavior)
- $\mathbb{1}[d_{\min} < d_{\text{unsafe}}]$: unsafe-distance penalty (discourages tailgating)
- $\mathbb{1}[\text{lane\_change}]$: lane-change penalty (reduces left–right oscillations)

---

## Hyperparameters (Main Ones)

Hyperparameters are kept in the config file (so we don’t have magic numbers inside logic).

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
- Early phase: low reward because the policy is basically random → frequent collisions
- Middle phase: reward rises as the agent learns “don’t crash immediately”
- Late phase: curve stabilizes; improvements become smaller (policy converges under current setup)

---

## Challenges & Fixes

### 1) Too many early crashes
At first, the agent often spammed `FASTER` and changed lanes aggressively, which caused lots of collisions.
We fixed this by:
- increasing collision penalty
- adding an unsafe-following penalty
- adding a small lane-change penalty

This made behavior noticeably smoother without making the agent slow.

---

### Generate evolution video

The evolution GIF is created by recording three agents under the same environment
configuration and random seed, then concatenating the videos.

#### Untrained (random policy)
python -m src.play --seed 0 --record

#### Half-trained
python -m src.play --seed 0 --record --model models/dqn_half.zip

#### Fully trained
python -m src.play --seed 0 --record --model models/dqn_full.zip

#### Merge into a single GIF
python -m src.make_evolution_gif
