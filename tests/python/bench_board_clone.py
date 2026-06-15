#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
B2 性能微基准：Board clone/restore 经 pyo3 的开销预算

测试场景：9×9 五子棋棋盘 × 400~800 MCTS simulations
每次 simulation 的环境操作 = 1 snapshot + 1 step + 1 restore

目标：估算单步 MCTS 中棋盘操作的时间开销，
判断是否在可接受范围内（< 1ms/sim → 800 sims < 0.8s）。
"""
import sys
import os
import time
import copy
import statistics

sys.stdout.reconfigure(encoding='utf-8')

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np


def create_board_9x9():
    """创建 9×9 五子棋 GomokuEnv，走几步让棋盘非空"""
    from tests.python.custom_envs.gomoku import GomokuEnv
    env = GomokuEnv(board_size=9, win_length=5, opponent='random')
    env.reset(seed=42)
    # 走 10 步让棋盘有些棋子（更真实的 MCTS 场景）
    for _ in range(5):
        valid = env.get_valid_actions()
        if valid and not env.done:
            env.step(np.random.choice(valid))
    return env


def bench_state_copy(env, n=10000):
    """测试 numpy state.copy() 的速度"""
    state = env.state
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        s = state.copy()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return times


def bench_state_restore(env, n=10000):
    """测试 state 赋值恢复的速度"""
    saved = env.state.copy()
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        env.state = saved.copy()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return times


def bench_deepcopy(env, n=10000):
    """测试 deepcopy 整个 env 的速度（备选方案对比）"""
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        _ = copy.deepcopy(env)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return times


def bench_step(env, n=5000):
    """测试单步 env.step() 的速度"""
    times = []
    for _ in range(n):
        env_copy = copy.deepcopy(env)
        valid = env_copy.get_valid_actions()
        if not valid or env_copy.done:
            env_copy.reset(seed=42)
            for _ in range(3):
                v = env_copy.get_valid_actions()
                if v and not env_copy.done:
                    env_copy.step(np.random.choice(v))
            valid = env_copy.get_valid_actions()

        action = np.random.choice(valid)
        t0 = time.perf_counter_ns()
        env_copy.step(action)
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return times


def bench_legal_mask(env, n=10000):
    """测试 get_valid_actions 的速度"""
    times = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        _ = env.get_valid_actions()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    return times


def bench_mcts_sim_cycle(env, n_sims=800):
    """模拟完整 MCTS simulation 周期：snapshot → step → restore
    
    这是最关键的测试——模拟 MCTS 单次 simulation 中的环境操作。
    """
    times = []
    for _ in range(n_sims):
        valid = env.get_valid_actions()
        if not valid or env.done:
            break
        action = np.random.choice(valid)

        t0 = time.perf_counter_ns()
        # 1. snapshot
        snap = env.state.copy()
        saved_to_play = env.to_play
        saved_done = env.done
        saved_last = env.last_action
        # 2. step (模拟展开)
        env.step(action)
        # 3. get legal mask (展开后需要知道合法动作)
        _ = env.get_valid_actions()
        # 4. restore
        env.state = snap
        env.to_play = saved_to_play
        env.done = saved_done
        env.last_action = saved_last
        t1 = time.perf_counter_ns()

        times.append(t1 - t0)
    return times


def report(name, times_ns):
    """打印统计报告"""
    times_us = [t / 1000 for t in times_ns]
    n = len(times_us)
    mean = statistics.mean(times_us)
    median = statistics.median(times_us)
    p95 = sorted(times_us)[int(n * 0.95)]
    p99 = sorted(times_us)[int(n * 0.99)]
    total_ms = sum(times_us) / 1000

    print(f"  {name}:")
    print(f"    n={n}  mean={mean:.1f}μs  median={median:.1f}μs  p95={p95:.1f}μs  p99={p99:.1f}μs")
    print(f"    total={total_ms:.1f}ms")
    return mean


def main():
    print("=" * 70)
    print("B2 性能微基准：9×9 五子棋 Board clone/restore")
    print("=" * 70)

    env = create_board_9x9()
    state_shape = env.state.shape
    state_bytes = env.state.nbytes
    print(f"\n棋盘: {env.board_size}×{env.board_size}")
    print(f"state shape: {state_shape}, dtype: {env.state.dtype}, bytes: {state_bytes}")
    print(f"当前棋面: 已走约 10 步（黑白各 ~5 步）")

    print("\n--- 单项操作基准 ---\n")

    t_copy = report("state.copy() [snapshot]", bench_state_copy(env, 10000))
    t_restore = report("state assign [restore]", bench_state_restore(env, 10000))
    t_legal = report("get_valid_actions [legal_mask]", bench_legal_mask(env, 10000))
    t_step = report("env.step() [单步]", bench_step(env, 3000))
    t_deep = report("deepcopy(env) [整体克隆，对比]", bench_deepcopy(env, 3000))

    print("\n--- MCTS 全周期模拟（snapshot → step → legal_mask → restore）---\n")

    # 重建干净的 env
    env = create_board_9x9()
    t_400 = bench_mcts_sim_cycle(env, 400)
    report("400 sims", t_400)

    env = create_board_9x9()
    t_800 = bench_mcts_sim_cycle(env, 800)
    report("800 sims", t_800)

    print("\n--- 预算评估 ---\n")

    avg_sim_us = statistics.mean([t / 1000 for t in t_800])
    for sims in [400, 800]:
        total_ms = avg_sim_us * sims / 1000
        print(f"  {sims} sims × {avg_sim_us:.1f}μs/sim = {total_ms:.1f}ms/move")

    print(f"\n  pyo3 额外开销估算（每次 call_method ~2-5μs）：")
    pyo3_overhead_per_sim = 4 * 5  # 4 calls × 5μs
    for sims in [400, 800]:
        py_total = avg_sim_us * sims / 1000
        pyo3_total = pyo3_overhead_per_sim * sims / 1000
        grand_total = py_total + pyo3_total
        print(f"  {sims} sims: Python={py_total:.1f}ms + pyo3≈{pyo3_total:.1f}ms = 总计≈{grand_total:.1f}ms/move")

    print(f"\n  神经网络推理（对比参考）：")
    print(f"    单次 9×9 前向传播（small CNN）≈ 0.5-2ms (CPU)")
    print(f"    {800} sims × 1ms/eval = 800ms（远大于环境开销）")

    threshold_ms = 100
    total_env_800 = avg_sim_us * 800 / 1000 + pyo3_overhead_per_sim * 800 / 1000
    verdict = "✅ 通过" if total_env_800 < threshold_ms else "❌ 超预算"
    print(f"\n  结论：800 sims 环境开销 ≈ {total_env_800:.1f}ms {verdict}（预算 < {threshold_ms}ms）")

    if total_env_800 < threshold_ms:
        print(f"  环境操作仅占单步总时间的 ~{total_env_800 / (total_env_800 + 800) * 100:.1f}%（网络推理才是瓶颈）")
    else:
        print(f"  ⚠️ 需要优化：考虑缩小棋盘 / 减少 sims / 批量调 Python")

    # 额外：对比 15×15（如果 9×9 通过，看 15×15 的增长）
    print("\n--- 附加：15×15 棋盘对比 ---\n")
    from tests.python.custom_envs.gomoku import GomokuEnv
    env15 = GomokuEnv(board_size=15, win_length=5, opponent='random')
    env15.reset(seed=42)
    for _ in range(5):
        v = env15.get_valid_actions()
        if v and not env15.done:
            env15.step(np.random.choice(v))

    print(f"  15×15 state: shape={env15.state.shape}, bytes={env15.state.nbytes}")
    report("15×15 state.copy()", bench_state_copy(env15, 5000))
    t_15_cycle = bench_mcts_sim_cycle(env15, 400)
    avg_15 = report("15×15 × 400 sims", t_15_cycle)
    print(f"  15×15 / 9×9 ratio ≈ {avg_15 / avg_sim_us:.1f}x")

    print("\n" + "=" * 70)
    print("基准测试完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
