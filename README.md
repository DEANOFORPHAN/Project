# CartPole DQN Training and Analysis System

This project is a full-stack Reinforcement Learning (RL) application for solving **CartPole-v1** with **Deep Q-Network (DQN)**.
It is designed for course project demonstration, reproducible experiments, and clear result visualization.

## 1. Project Overview

This system combines:

- RL model training (DQN, PyTorch, Gymnasium)
- Data output pipeline (metrics, rewards, plots, checkpoints)
- Backend APIs (FastAPI)
- Frontend dashboard (React + Vite)

Current project status:

- DQN training pipeline is implemented and runnable
- Best model checkpoint and evaluation flow are implemented
- Dashboard can display metrics, reward curve, evaluation table, and play animation

## 2. Core Features

- Train a DQN agent for CartPole-v1
- Save training outputs automatically
- Evaluate best model over multiple episodes
- Render reward curve image
- Play one greedy CartPole episode as GIF animation
- Visualize all key results in a web dashboard

## 3. Tech Stack

- Backend: Python, FastAPI, Uvicorn
- RL: PyTorch, Gymnasium (CartPole-v1)
- Data/Visualization: NumPy, Matplotlib
- Media: imageio (GIF generation)
- Frontend: React, Vite, JavaScript, CSS

## 4. Project Structure

```text
project/
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── models/
│   │   │   ├── dqn_model.py
│   │   │   └── replay_buffer.py
│   │   ├── routes/
│   │   │   └── rl_routes.py
│   │   ├── services/
│   │   │   ├── train_service.py
│   │   │   ├── evaluate_service.py
│   │   │   └── play_service.py
│   │   └── outputs/
│   │       ├── best_model.pth
│   │       ├── rewards.csv
│   │       ├── metrics.json
│   │       ├── reward_curve.png
│   │       └── play_latest.gif
│   └── testRL/
│       ├── test_train.py
│       └── test_evaluate.py
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── App.css
│       └── ...
├── requirements.txt
├── HYPERPARAMETERS.md
└── README.md
```

## 5. Setup and Run

### 5.1 Create and activate environment

```bash
conda create -n cartpole python=3.10 -y
conda activate cartpole
```

### 5.2 Install dependencies

```bash
cd /Users/xufeixiang/Desktop/Docs/CUHK/TERM2/dataScience/Project
pip install -r requirements.txt
```

### 5.3 Run backend

```bash
cd backend
uvicorn app.main:app --reload
```

Backend docs:

- [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5.4 Run frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend page (default):

- [http://localhost:5173](http://localhost:5173)

## 6. API Endpoints

- `GET /api/metrics`  
  Return training metrics from `metrics.json`.

- `POST /api/evaluate`  
  Evaluate `best_model.pth` over 20 episodes.

- `GET /api/plot`  
  Return `reward_curve.png`.

- `POST /api/play`  
  Run one greedy episode and generate `play_latest.gif`.

- `GET /api/play/gif`  
  Return the latest generated play GIF.

## 7. Training and Evaluation

### Train

```bash
cd backend
python -m testRL.test_train
```

### Evaluate

```bash
cd backend
python -m testRL.test_evaluate
```

## 8. Output Files

After training, outputs are stored in `backend/app/outputs/`:

- `best_model.pth`: best checkpoint model
- `rewards.csv`: reward per episode
- `metrics.json`: summary metrics + hyperparameters
- `reward_curve.png`: reward and moving-average curve
- `play_latest.gif`: latest play animation

## 9. Hyperparameters

Current stable hyperparameter set is documented in:

- `HYPERPARAMETERS.md`

Main config source:

- `backend/app/core/config.py`

## 10. Notes and Troubleshooting

- If `imageio` is missing, install dependencies again:
  - `pip install -r requirements.txt`
- If CartPole render fails with pygame error:
  - ensure `gymnasium[classic-control]` is installed (already included in `requirements.txt`)
- If frontend cannot access backend:
  - confirm backend is running on `127.0.0.1:8000`
  - confirm frontend runs on `localhost:5173` or update CORS settings
