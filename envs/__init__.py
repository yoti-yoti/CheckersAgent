from gymnasium.envs.registration import register

register(
    id="Checkers-v0",
    entry_point="envs.checkers_env:CheckersEnv",
)
