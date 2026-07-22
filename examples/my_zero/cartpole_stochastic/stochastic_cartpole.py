"""CartPole with random velocity perturbation for Stochastic MuZero validation.

注册为 gymnasium 环境 ``StochasticCartPole-v0``，可通过
``gymnasium.make("StochasticCartPole-v0")`` 创建。
"""
import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv


class StochasticCartPoleEnv(CartPoleEnv):
    """CartPole with random velocity noise after each step.

    继承 CartPoleEnv 并在 step 后对 cart_velocity 和 pole_angular_velocity
    注入均匀随机扰动，模拟 agent 无法控制的环境随机性。
    """

    metadata = {**CartPoleEnv.metadata, "name": "StochasticCartPole-v0"}

    def __init__(self, noise_scale=0.05, **kwargs):
        super().__init__(**kwargs)
        self.noise_scale = noise_scale

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        noise = self.np_random.uniform(
            -self.noise_scale, self.noise_scale, size=2
        )
        obs = np.array(obs, dtype=np.float32)
        obs[1] += noise[0]  # cart velocity
        obs[3] += noise[1]  # pole angular velocity
        self.state = (obs[0], obs[1], obs[2], obs[3])
        return obs, reward, terminated, truncated, info


gym.register(
    id="StochasticCartPole-v0",
    entry_point="stochastic_cartpole:StochasticCartPoleEnv",
    max_episode_steps=500,
)
