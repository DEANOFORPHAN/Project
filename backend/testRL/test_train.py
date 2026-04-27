from app.services.train_service import train_dqn

if __name__ == "__main__":
    result = train_dqn()
    print(result["episode_rewards"])