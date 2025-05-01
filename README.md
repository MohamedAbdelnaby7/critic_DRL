# RL Agents for LunarLander (DQN & DDPG)

This project trains reinforcement learning agents using DQN (for discrete control) and DDPG (for continuous control) on the LunarLander environments from Gymnasium.

------------------------------------------------------------

🔧 Installation Guide

Set up a clean Python environment and install all required packages to run the training scripts for DQN and DDPG.

📉 Step-by-Step Instructions

1. Create and activate a Conda environment:

```bash
conda create -n rl_env python=3.10 -y
conda activate rl_env
```

2. Install PyTorch:

CPU-only:
```bash
pip install torch torchvision torchaudio
```

GPU with CUDA 11.8:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Create a requirements.txt file with the following content:

```txt
gymnasium[box2d]~=0.29.1
numpy
tensorboard
matplotlib
box2d
pygame
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

4. Verify the installation:

```bash
python -c "import gymnasium; env = gymnasium.make('LunarLander-v2', render_mode='human'); env.reset(); print('\u2713 Gymnasium works!')"
```

------------------------------------------------------------

🚀 Running the Agents

Train the DQN agent (discrete control):

```bash
python dqn_lunarlander.py
```

Train the DDPG agent (continuous control):

```bash
python ddpg_lunarlander.py
```

------------------------------------------------------------

📊 Monitor Training with TensorBoard

```bash
tensorboard --logdir runs/
```

Then open this in your browser:

http://localhost:6006

------------------------------------------------------------

📁 Suggested Project Structure

```
rl-lunarlander/
├── dqn_lunarlander.py
├── ddpg_lunarlander.py
├── requirements.txt
├── runs/              # TensorBoard logs
├── models/            # Saved checkpoints
└── README.md
```

------------------------------------------------------------

❓ Common Issues

Problem: pygame is not installed  
Solution: Run `pip install pygame`

Problem: swig.exe failed during install  
Solution: Use `box2d` instead of `box2d-py` (already in requirements.txt)

Problem: gymnasium.make(...) fails  
Solution: Make sure you installed `gymnasium[box2d]` and `box2d`

Problem: CUDA not detected in PyTorch  
Solution: Run `python -c "import torch; print(torch.cuda.is_available())"`

------------------------------------------------------------

📄 Clean Uninstall (optional)

```bash
conda deactivate
conda remove -n rl_env --all
```

------------------------------------------------------------

✨ License

This project is open-source under the MIT License.

