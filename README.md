# Deep Reinforcement Learning for Traffic Signal Control

A DQN and Fluid-Dynamic Simulation Framework with Adam and PSO Optimizer Comparison

This repository implements a **Deep Reinforcement Learning (DRL)**–based adaptive traffic signal controller using **Deep Q-Networks (DQN)** combined with a **fluid-dynamic traffic simulation** built with Taichi.
It also includes a detailed comparison between **Adam** and **Particle Swarm Optimization (PSO)** for training RL agents.

---

## Key Features

* DQN-based adaptive traffic signal control
* Fluid-density traffic simulation using Taichi
* Real-time traffic light switching to reduce congestion
* Optimizer comparison: Adam vs PSO
* Visualization support (heatmaps, congestion plots, simulation outputs)
* Modular RL framework (`rl_agent/`, `simulation/`, `data_processing.py`)
* Fully reproducible experiments and evaluation scripts

---

## Project Structure

```
SEAI project/
│── data/
│   ├── ngsim_density.csv
│   └── (Download NGSIM dataset manually – see instructions below)
│
│── rl_agent/
│   ├── agent.py
│   ├── adam_agent.py
│   ├── environment.py
│   └── __init__.py
│
│── simulation/
│   ├── simulation_core.py
│   └── __init__.py
│
│── data_processing.py
│── main.py
│── traffic_gui_with_rl.py
│── test_agent.py
```

---

# Dataset Access (Important)

This project uses the **NGSIM Vehicle Trajectory Dataset**, which is approximately 1.4 GB and exceeds GitHub’s 100 MB file size limit.
It must be downloaded manually.

### Download Link

Official dataset page:
[https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)

Required file:

* `Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250318.csv`

Place it in the `data/` directory:

```
data/
│── Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250318.csv
│── ngsim_density.csv
```

---

# Installation

### 1. Clone the repository

```bash
git clone https://github.com/sonal2803/traffic-signal-drl.git
cd traffic-signal-drl
```

### 2. Install dependencies

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

Otherwise, install manually:

```bash
pip install numpy pandas taichi torch matplotlib opencv-python
```

---

# Running the Simulation

### 1. Preprocess the dataset

```bash
python data_processing.py
```

### 2. Train the DQN agent

```bash
python main.py
```

### 3. Visualize or test the trained agent

```bash
python traffic_gui_with_rl.py
```

or

```bash
python test_agent.py
```

---

# DRL Approach

Traffic is modeled as a **fluid density map**, and the agent learns optimal signal switching policies.

The agent learns to:

* Switch signals at the right time
* Maintain green phases when beneficial
* Optimize traffic flow dynamically

Reward is based on:

* Queue length reduction
* Density minimization
* Overall throughput improvement

---

# Adam vs PSO Optimizer Study

A complete evaluation of Adam and PSO optimizers was conducted for DQN training.

| Optimizer | Convergence Speed | Stability   | Reward Score |
| --------- | ----------------- | ----------- | ------------ |
| Adam      | Faster            | More stable | Higher       |
| PSO       | Slower            | Oscillatory | Lower        |

**Conclusion:**
Adam consistently outperforms PSO for DQN-based traffic control across all test scenarios.

---

# Results Summary

* Significant reduction in congestion
* Higher throughput
* Smoother phase transitions
* Clear visualization of agent learning behavior
* Superior performance using Adam

### Future Enhancements

* Multi-agent reinforcement learning
* Integration of real-time sensor data
* DDPG-based continuous control
* Graph neural network (GNN)–based traffic modeling

---

# Author

**Sonal Panda**
B.Tech CSE (AIML), SRM Institute of Science and Technology
Email: **[sonalpanda28@gmail.com](mailto:sonalpanda28@gmail.com)**
GitHub: **[https://github.com/sonal2803](https://github.com/sonal2803)**

