import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.services.evaluate_service import evaluate_dqn
from app.services.play_service import play_once_and_save_gif

router = APIRouter(tags=["rl"])


@router.get("/api/metrics")
def get_metrics():
    """Read and return training metrics from outputs/metrics.json."""
    metrics_path = Path(__file__).resolve().parents[1] / "outputs" / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail=f"Metrics file not found: {metrics_path}")

    try:
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail="metrics.json is invalid JSON") from exc

    return metrics


@router.post("/api/evaluate")
def evaluate_agent():
    """Evaluate best_model.pth and return evaluation results."""
    try:
        return evaluate_dqn(evaluation_episodes=20)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/plot")
def get_reward_plot():
    """Return reward curve image generated during training."""
    plot_path = Path(__file__).resolve().parents[1] / "outputs" / "reward_curve.png"
    if not plot_path.exists():
        raise HTTPException(status_code=404, detail=f"Plot file not found: {plot_path}")
    return FileResponse(plot_path, media_type="image/png", filename="reward_curve.png")


@router.post("/api/play")
def play_agent_once():
    """Run one greedy episode and generate latest CartPole GIF."""
    try:
        result = play_once_and_save_gif(max_steps=500, fps=30)
        return {
            "message": "Play animation generated successfully.",
            "reward": result["reward"],
            "steps": result["steps"],
            "frames": result["frames"],
            "gif_url": "/api/play/gif",
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/play/gif")
def get_play_gif():
    """Return latest generated play animation GIF."""
    gif_path = Path(__file__).resolve().parents[1] / "outputs" / "play_latest.gif"
    if not gif_path.exists():
        raise HTTPException(status_code=404, detail=f"Play GIF not found: {gif_path}")
    return FileResponse(gif_path, media_type="image/gif", filename="play_latest.gif")
