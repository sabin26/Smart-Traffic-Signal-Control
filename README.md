# Smart Traffic Signal Control in Kathmandu

An adaptive traffic signal control system that compares reinforcement learning approaches with baseline controllers. This project uses **SUMO (Simulation of Urban MObility)** with real road networks from OpenStreetMap to simulate and optimize traffic flow in Kathmandu.

---

## 📋 Project Overview

This project implements and compares multiple traffic signal control strategies:

### Baseline Controllers

-   **Fixed-Time Controller:** Traditional pre-timed signal phases
-   **Max Pressure Controller:** Pressure-based optimization without learning
-   **Supervised MLP:** Neural network trained on labeled data

### Reinforcement Learning Agents

-   **PPO (Proximal Policy Optimization):** With multiple optimizers (Adam, SGD, RMSprop)
-   **DQN (Deep Q-Network):** Value-based learning approach
-   **A2C (Advantage Actor-Critic):** Policy gradient with value function

---

## 🏗️ Project Structure

```
smart_traffic_signal_control/
├── README.md
├── smart_traffic_signal_control.py        # Main application (Marimo notebook)
└── sumo/
   └── kathmandu/
       ├── osm.sumocfg.xml               # SUMO simulation configuration
       ├── osm.netccfg.xml               # Network definition
       ├── osm.*.trips.xml               # Traffic demand files
```

---

## 🚀 Features

-   **SUMO Integration:** Realistic traffic simulation with TraCI interface
-   **Multiple Vehicle Types:** Passenger cars, motorcycles, buses, trucks, bicycles
-   **Parallel Training:** Concurrent training of multiple agents
-   **Real Network Topology:** Road network data from OpenStreetMap
-   **Comprehensive Evaluation:** Metrics include reward, queue length, waiting time, and throughput
-   **Reproducible Results:** Fixed random seed (42) for consistency
-   **GPU Support:** Automatic CUDA detection for accelerated training

---

## 🔧 Requirements

-   Python 3.12+
-   PyTorch
-   NumPy
-   SUMO (with TraCI Python bindings)
-   Marimo (for interactive notebook)
-   OpenStreetMap OSM file for network generation

### Installation

1. **Set up Python environment:**

    ```bash
    python -m venv ann-venv
    source ann-venv/bin/activate  # On macOS/Linux
    ```

2. **Install dependencies:**

    ```bash
    pip install torch numpy marimo
    ```

3. **Install SUMO:**

    - Follow [SUMO Installation Guide](https://sumo.dlr.de/docs/Installing/index.html)
    - Set `SUMO_HOME` environment variable:
        ```bash
        export SUMO_HOME=/path/to/sumo
        ```

4. **Verify setup:**
    ```bash
    echo $SUMO_HOME
    which sumo
    which sumo-gui
    ```

---

## 📊 Configuration

Key hyperparameters are defined in the `Config` class:

```python
# Environment Settings
max_steps_per_episode: int = 300      # Duration of each episode
green_duration: int = 30              # Default green phase (seconds)
max_vehicles_per_lane: int = 20       # Queue capacity

# RL Training
num_episodes: int = 100               # Episodes per training run
batch_size: int = 512                 # PPO batch size
learning_rate: float = 3e-4           # Learning rate

# Model-specific
PPO clip_epsilon: float = 0.25        # Clipping parameter
DQN buffer_size: int = 10000          # Experience replay size
```

---

## 🎯 Running the Application

### Option 1: Interactive Marimo Notebook

```bash
marimo edit smart_traffic_signal_control.py
```

### Option 2: Run Directly

```bash
python smart_traffic_signal_control.py
```

### With GUI (SUMO visualization)

Modify the environment initialization:

```python
env = SUMOTrafficEnv(config, gui=True, ...)
```

---

## 📈 Key Metrics

The system evaluates each controller on:

| Metric               | Description                                  |
| -------------------- | -------------------------------------------- |
| **Mean Reward**      | Average cumulative reward per episode        |
| **Avg Queue Length** | Average vehicles waiting at intersections    |
| **Avg Waiting Time** | Average seconds vehicles spend waiting       |
| **Throughput**       | Number of vehicles that completed their trip |
| **Episode Length**   | Steps taken per training episode             |

---

## 🧠 Agent Architecture

### State Space (12-dimensional)

-   Queue lengths for 4 lanes
-   Waiting times for 4 lanes
-   One-hot encoded current phase (4 dimensions)

### Action Space (4-dimensional)

Four possible signal phase transitions (N-S green, E-W green, etc.)

### Network Architecture

-   Hidden layers: 128 neurons
-   Activation: ReLU
-   Output layer: Value and policy heads (for PPO/A2C) or Q-values (for DQN)

---

## 📂 Output Files

Training generates multiple output files:

-   `*_edgeData.xml`: Aggregated edge statistics (vehicles, speed, etc.)
-   `*_stopinfos.xml`: Stop sign data
-   `*_tripinfos.xml`: Trip statistics (departure, arrival, duration)
-   `*_stats.xml`: Overall simulation statistics

---

## 🔬 Experimental Setup

-   **Random Seed:** 42 (reproducible results)
-   **Device:** Auto-detects GPU; falls back to CPU
-   **Evaluation:** 5 independent episodes per controller
-   **Total Training:** 100 episodes × multiple controllers
-   **Parallel Execution:** Multi-threaded training when available

---

## 📝 Notebook Structure

The main file is organized as an interactive Marimo notebook with sections:

1. **Title & Setup** - Project introduction and imports
2. **Configuration** - Hyperparameter definitions
3. **SUMO Environment** - Traffic simulation wrapper
4. **Baseline Controllers** - Fixed-time, Max Pressure, MLP
5. **RL Agents** - PPO, DQN, A2C implementations
6. **Training Loop** - Multi-agent concurrent training
7. **Evaluation** - Performance metrics and comparison
8. **Results & Analysis** - Summary tables and findings
9. **Conclusions** - Key takeaways

---

## ⚙️ Environmental Setup

Ensure these environment variables are set:

```bash
export SUMO_HOME=/path/to/sumo
export PYTHONPATH=$PYTHONPATH:/path/to/project
```

---

## 🐛 Troubleshooting

### SUMO Connection Errors

```
FileNotFoundError: SUMO config not found
```

**Solution:** Verify the SUMO configuration file path is correct relative to your working directory.

### PROJ Library Warnings

These are suppressed by the `SuppressStderr` context manager in the code.

### Out of Memory

Reduce `batch_size`, `buffer_size`, or number of parallel environments.

### GPU Not Available

The code automatically falls back to CPU. For CUDA support, ensure PyTorch is built with CUDA.

---

## 📚 References

-   SUMO Documentation: https://sumo.dlr.de/docs/
-   PyTorch Documentation: https://pytorch.org/docs/
-   Marimo: https://marimo.io/
-   PPO Paper: https://arxiv.org/abs/1707.06347
-   DQN Paper: https://arxiv.org/abs/1312.5602
-   A2C Details: https://arxiv.org/abs/1602.01783

---

## 📄 License

Academic project for Artificial Neural Network (ANN) coursework.
