#!/usr/bin/env bash
# seed 级进程并行跑 Gomoku M3 消融臂（性能台账候选 #8「便宜档」）。
#
# 原理：一个臂的 N 个 seed 是互不通信的独立实验，各自起一个 OS 进程同时跑
# （载体内 M3_SEEDS 环境变量过滤），臂墙钟 ÷N；每个 seed 的数值轨迹与串行跑
# 逐 bit 一致。各进程日志独立落盘，全部成功后按 seed 序拼接成单一臂级日志
# （格式与串行产物一致），并在末尾追加合并汇总块。
#
# 用法：
#   bash scripts/bench_m3_seedpar.sh <test_filter> [seeds] [out_log]
#     test_filter  载体测试名（如 gomoku_pure_selfplay_per）
#     seeds        空格分隔，默认 "42 43 44"
#     out_log      合并日志路径，默认 .bench/<test_filter>_<yyyymmdd>.log
#   BLAS_FLAG 环境变量可覆盖 cargo feature（默认 "--features blas-mkl"；
#   just bench-m3-seedpar 会传入自动检测结果）。
#
# 口径注记：并行跑时各 seed 共享机器（缓存/内存带宽竞争），日志中的
# t=/wall= 计时会比独占机器略大；env_steps 与全部学习指标不受影响
# （wall-clock 本就不是评价指标，纲领 §2.2）。
set -euo pipefail

FILTER=${1:?用法: bash scripts/bench_m3_seedpar.sh <test_filter> [seeds] [out_log]}
SEEDS=${2:-"42 43 44"}
cd "$(dirname "$0")/.."
OUT=${3:-".bench/${FILTER}_$(date +%Y%m%d).log"}
# 无冒号形式：仅在变量未设置时用默认（just 检测到纯 Rust 时会传入空串，须尊重）
BLAS_FLAG=${BLAS_FLAG-"--features blas-mkl"}

if [ -e "$OUT" ]; then
    echo "[seed-par] 输出日志已存在：$OUT（拒绝覆盖，请显式指定第三参数）" >&2
    exit 1
fi

# 1) 预编译（一次编译、三进程复用同一二进制，避免并发 cargo 撞构建锁）
echo "[seed-par] 预编译 lib 测试二进制（$BLAS_FLAG）..."
# shellcheck disable=SC2086
cargo test --release $BLAS_FLAG --lib --no-run

# 2) 定位测试二进制（message-format=json 精确取 lib test 目标，避免猜 hash）
# shellcheck disable=SC2086
EXE=$(cargo test --release $BLAS_FLAG --lib --no-run --message-format=json 2>/dev/null | python -c "
import json, sys
exe = ''
for line in sys.stdin:
    try:
        m = json.loads(line)
    except ValueError:
        continue
    if m.get('reason') != 'compiler-artifact' or not m.get('executable'):
        continue
    target, profile = m.get('target', {}), m.get('profile', {})
    if target.get('name') == 'only_torch' and 'lib' in target.get('kind', []) and profile.get('test'):
        exe = m['executable']
print(exe)
")
[ -n "$EXE" ] || { echo "[seed-par] 未定位到 lib 测试二进制" >&2; exit 1; }
command -v cygpath >/dev/null 2>&1 && EXE=$(cygpath -u "$EXE")
echo "[seed-par] 测试二进制：$EXE"

# 3) 每 seed 一个进程并行跑，日志分文件
declare -a PIDS=() LOGS=() SEED_ARR=()
for seed in $SEEDS; do
    log="${OUT%.log}_seed${seed}.log"
    rm -f "$log"
    echo "[seed-par] 启动 seed=$seed → $log"
    M3_SEEDS=$seed "$EXE" "$FILTER" --ignored --nocapture --test-threads=1 \
        > "$log" 2>&1 &
    PIDS+=($!); LOGS+=("$log"); SEED_ARR+=("$seed")
done
echo "[seed-par] ${#PIDS[@]} 个 seed 进程已并行启动（观察进度: tail -f ${LOGS[*]}）"

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "[seed-par] seed=${SEED_ARR[$i]} 完成"
    else
        echo "[seed-par] seed=${SEED_ARR[$i]} 失败（exit≠0），保留 ${LOGS[$i]} 以供排查" >&2
        FAIL=1
    fi
done
if [ "$FAIL" -ne 0 ]; then
    echo "[seed-par] 存在失败 seed，不合并日志" >&2
    exit 1
fi

# 4) 按 seed 序拼接为臂级日志 + 追加合并汇总块，然后清理分 seed 文件
{
    echo "# seed 并行跑合并日志：arm=$FILTER seeds=($SEEDS) 生成于 $(date '+%F %T')"
    echo "# 注意：并行跑的 t=/wall= 计时含资源竞争，略大于独占机器；env_steps 与学习指标不受影响"
    for i in "${!LOGS[@]}"; do
        echo
        echo "===== seed ${SEED_ARR[$i]} ====="
        cat "${LOGS[$i]}"
    done
    echo
    echo "--- M3 arm=$FILTER 汇总（seed 并行合并，base 对照见 bench 头注释）---"
    grep -h '^  seed=' "${LOGS[@]}"
} > "$OUT"
rm -f "${LOGS[@]}"

echo "[seed-par] 完成，合并日志：$OUT"
echo "[seed-par] 汇总："
grep '^  seed=' "$OUT" | tail -n "$(echo "$SEEDS" | wc -w)"
