# Only-Torch Justfile
# 使用: just <command>

# 默认命令：运行测试和 lint
default: test lint

# 完整检查：清理、测试、lint、所有 examples
all: clean test lint examples

# ==================== 测试 ====================

# 运行常规单元测试（跳过 #[ignore] 标记的测试）
test:
    cargo test

# 运行全量测试（包含 #[ignore] 标记的联网等测试）
test-all:
    cargo test -- --include-ignored

# 仅运行 #[ignore] 标记的测试（联网等特殊测试）
test-ignored:
    cargo test -- --ignored

# 运行所有单元测试（单线程，用于调试）
test-serial:
    cargo test -- --test-threads=1

# 运行特定测试（用法: just test-filter <pattern>）
test-filter pattern:
    cargo test {{pattern}}

# ==================== Examples ====================

# 运行所有 examples（含 RL）
examples: example-xor example-iris example-sine example-mnist example-mnist-cnn example-mnist-gan example-california example-parity example-dual-input example-siamese example-dual-output example-multi-io example-multi-label example-cartpole-sac example-pendulum-sac example-moving-sac

# 运行所有 parity examples（RNN/LSTM/GRU）
example-parity: example-parity-fixed example-parity-var example-parity-lstm example-parity-gru

# 单个 examples
example-xor:
    @echo "=== Running XOR ==="
    cargo run --example xor

example-iris:
    @echo "=== Running Iris ==="
    cargo run --example iris

example-sine:
    @echo "=== Running Sine Regression ==="
    cargo run --example sine_regression

example-mnist:
    @echo "=== Running MNIST ==="
    cargo run --example mnist

example-mnist-cnn:
    @echo "=== Running MNIST CNN ==="
    cargo run --example mnist_cnn

example-mnist-gan:
    @echo "=== Running MNIST GAN ==="
    cargo run --example mnist_gan

example-california:
    @echo "=== Running California Housing ==="
    cargo run --example california_housing

example-parity-fixed:
    @echo "=== Running Parity (RNN, Fixed Length) ==="
    cargo run --example parity_rnn_fixed_len

example-parity-var:
    @echo "=== Running Parity (RNN, Variable Length) ==="
    cargo run --example parity_rnn_var_len

example-parity-lstm:
    @echo "=== Running Parity (LSTM, Variable Length) ==="
    cargo run --example parity_lstm_var_len

example-parity-gru:
    @echo "=== Running Parity (GRU, Variable Length) ==="
    cargo run --example parity_gru_var_len

example-dual-input:
    @echo "=== Running Dual Input Add (forward2) ==="
    cargo run --example dual_input_add

example-siamese:
    @echo "=== Running Siamese Similarity (shared encoder) ==="
    cargo run --example siamese_similarity

example-dual-output:
    @echo "=== Running Dual Output Classify (multi-output) ==="
    cargo run --example dual_output_classify

example-multi-io:
    @echo "=== Running Multi IO Fusion (multi-input + multi-output) ==="
    cargo run --example multi_io_fusion

example-multi-label:
    @echo "=== Running Multi Label Point (BCE Loss) ==="
    cargo run --example multi_label_point

example-cartpole-sac:
    @echo "=== Running CartPole SAC (requires Python + gymnasium) ==="
    cargo run --example cartpole_sac

example-pendulum-sac:
    @echo "=== Running Pendulum SAC (requires Python + gymnasium) ==="
    cargo run --example pendulum_sac

example-moving-sac:
    @echo "=== Running Moving Hybrid SAC (requires Python + gymnasium) ==="
    cargo run --example moving_sac

# ==================== 代码质量 ====================

# 运行 clippy lint
lint:
    cargo clippy

# 运行 clippy 并自动修复
lint-fix:
    cargo clippy --fix --allow-dirty --allow-staged

# 格式化代码
fmt:
    cargo fmt

# 检查格式（不修改）
fmt-check:
    cargo fmt -- --check

# ==================== 构建 ====================

# Debug 构建
build:
    cargo build

# Release 构建
build-release:
    cargo build --release

# 检查编译（不生成二进制）
check:
    cargo check

# ==================== 清理 ====================

# 清理构建产物
clean:
    cargo clean

# 完整清理并更新依赖
clean-all:
    cargo clean && cargo update

# ==================== Python 测试 ====================

# 运行所有 Gymnasium 环境测试
py-gym:
    @echo "=== Running All Gymnasium Tests ==="
    python tests/python/gym/run_all_tests.py

# 运行单个 gym 测试
py-gym-basic:
    @echo "=== Running Basic Discrete ==="
    python tests/python/gym/test_01_basic_discrete.py

py-gym-continuous:
    @echo "=== Running Basic Continuous ==="
    python tests/python/gym/test_02_basic_continuous.py

py-gym-box2d:
    @echo "=== Running Box2D ==="
    python tests/python/gym/test_03_box2d.py

py-gym-mujoco:
    @echo "=== Running MuJoCo ==="
    python tests/python/gym/test_04_mujoco.py

py-gym-atari:
    @echo "=== Running Atari ==="
    python tests/python/gym/test_05_atari.py

py-gym-minari:
    @echo "=== Running Minari ==="
    python tests/python/gym/test_06_minari.py

py-gym-hybrid:
    @echo "=== Running Hybrid Action Space ==="
    python tests/python/gym/test_07_hybrid.py

py-gym-gomoku:
    @echo "=== Running Gomoku Custom Environment ==="
    python tests/python/gym/test_08_gomoku.py

# ==================== 文档 ====================

# 生成文档
doc:
    cargo doc --no-deps

# 生成并打开文档
doc-open:
    cargo doc --no-deps --open
