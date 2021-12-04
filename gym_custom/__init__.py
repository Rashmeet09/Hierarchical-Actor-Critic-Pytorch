from gym_custom.MountainCarContinuous import Continuous_MountainCarEnv
from gym_custom.Pendulum import PendulumEnv
from gym.envs.registration import register

register(
    id="MountainCarContinuous-v1",
    entry_point="gym_custom:Continuous_MountainCarEnv",
)

register(
    id="Pendulum-v1",
    entry_point="gym_custom:PendulumEnv",
)
