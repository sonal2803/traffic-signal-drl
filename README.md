# 🚦 Deep Reinforcement Learning for Traffic Signal Control  
A DQN + Fluid Dynamic Simulation Framework with Adam and PSO Optimizer Comparison

This repository implements a **Deep Reinforcement Learning (DRL)**–based adaptive traffic signal controller using **Deep Q-Networks (DQN)** and a **fluid-dynamic inspired traffic simulation** built with Taichi.  
The project also includes a detailed comparison between **Adam** and **Particle Swarm Optimization (PSO)** for training DRL agents.

---

## 🔥 Key Features

- **Deep Q-Network (DQN)**–based traffic control  
- **Fluid-density traffic simulation** using Taichi  
- **Adaptive traffic light switching** to minimize congestion  
- **Optimizer comparison:** Adam vs PSO  
- **Visualization:** heatmaps, congestion plots, simulation GIFs  
- Modular RL framework (`rl_agent/`, `simulation/`, `data_processing.py`)  
- Reproducible experiments and evaluation scripts  

---

## 📁 Project Structure

SEAI project/
│── data/
│ ├── ngsim_density.csv
│ └── (You must download NGSIM dataset manually – see below)
│
│── rl_agent/
│ ├── agent.py
│ ├── adam_agent.py
│ ├── environment.py
│ └── init.py
│
│── simulation/
│ ├── simulation_core.py
│ └── init.py
│
│── data_processing.py
│── main.py
│── traffic_gui_with_rl.py
│── test_agent.py


---

# 📊 Dataset Access (IMPORTANT)

This project uses the **NGSIM vehicle trajectory dataset**, which is too large for GitHub (1.4 GB).  
GitHub's file size limit is **100 MB**, so the dataset must be downloaded manually.

### 📥 Download Dataset
Official dataset link:  
🔗 https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm

You must download the file:

- **Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250318.csv**

and place it here:


Directory will look like:

data/
│── Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_20250318.csv
│── ngsim_density.csv


---

# 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/sonal2803/traffic-signal-drl.git
cd traffic-signal-drl
### INSTALL DEPENDENCIES 

pip install -r requirements.txt
If there is no requirements file, install manually:

pip install numpy pandas taichi torch matplotlib opencv-python

▶️ Running the Simulation
1. Preprocess data
python data_processing.py

2. Train the DQN agent
python main.py

3. Visualize or test the agent
python traffic_gui_with_rl.py


or

python test_agent.py

🧠 DRL Approach

The project models traffic as a fluid density heatmap, and the agent learns:

When to switch signals

When to keep signals

How to optimize flow in real time

Reward is based on:

Queue lengths

Density reduction

Throughput

⚖️ Adam vs PSO Optimizer Study

This project includes a comparative study of:

Optimizer	Convergence Speed	Stabilization	Reward Score
Adam	Faster	More stable	Higher
PSO	Slower	Oscillatory	Lower

Result:
➡️ Adam outperforms PSO for DQN-based traffic control in almost all scenarios.

📈 Results Summary

Significant reduction in congestion metrics

Improved average throughput

Smoother traffic phase transitions

Clear visualization of agent learning

Adam consistently performs better than PSO

Future extensions:

Multi-agent RL

Real-time sensor integration

Deep deterministic policy gradients (DDPG)

Graph neural network traffic modeling

👤 Author

Sonal Panda
B.Tech CSE (AIML), SRM Institute of Science and Technology
📧 sonalpanda28@gmail.com

🔗 GitHub: https://github.com/sonal2803
