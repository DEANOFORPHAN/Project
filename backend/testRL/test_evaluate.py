from app.services.evaluate_service import evaluate_dqn


if __name__ == "__main__":
    result = evaluate_dqn(evaluation_episodes=20)

    print("Evaluation finished.")
    print(f"evaluation_episodes: {result['evaluation_episodes']}")
    print(f"average_reward: {result['average_reward']:.2f}")
    print(f"best_reward: {result['best_reward']:.2f}")
    print(f"rewards: {result['rewards']}")
