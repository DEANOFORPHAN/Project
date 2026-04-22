import gymnasium as gym

env = gym.make("CartPole-v1")
obs, _ = env.reset()
print(obs)