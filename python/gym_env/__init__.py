"""
only_torch 自定义 Gymnasium 环境

导入此模块时自动注册所有自定义环境。
安装：pip install -e python/gym_env
使用：import gym_env; gymnasium.make("Gomoku-selfplay-v0")
"""
from gymnasium.envs.registration import register

# ============================================================================
# 五子棋环境注册
# ============================================================================

# Self-play 版（无内置对手，双方外部落子；self-play 自对弈训练用）
register(
    id="Gomoku-selfplay-v0",
    entry_point="gym_env.gomoku.env:GomokuSelfPlayEnv",
    kwargs={"board_size": 9, "win_length": 5},
    max_episode_steps=81,  # 9×9 棋盘最多 81 步
)

# 带 naive 对手版（评测 / 人机用）
_gomoku_levels = ["random", "naive0", "naive1", "naive2", "naive3"]

for _level in _gomoku_levels:
    for _size, _max_steps in [(9, 81), (15, 225)]:
        _suffix = "" if _size == 9 else f"-{_size}x{_size}"
        register(
            id=f"Gomoku-{_level}{_suffix}-v0",
            entry_point="gym_env.gomoku.env:GomokuEnv",
            kwargs={
                "board_size": _size,
                "win_length": 5,
                "opponent": _level,
            },
            max_episode_steps=_max_steps,
        )
