# Only-Torch Justfile
# 使用: just <command>

# 默认命令：运行测试和 lint
default: test lint

# 完整检查：清理、测试、lint、所有 examples
all: clean test lint examples

# ==================== 测试 ====================

# 运行所有单元测试
test:
    cargo test

# 运行所有单元测试（单线程，用于调试）
test-serial:
    cargo test -- --test-threads=1

# 运行特定测试（用法: just test-filter <pattern>）
test-filter pattern:
    cargo test {{pattern}}

# ==================== Examples ====================

# 运行所有 examples
examples: example-xor example-iris example-sine example-mnist example-mnist-gan example-california example-parity

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

# ==================== 文档 ====================

# 生成文档
doc:
    cargo doc --no-deps

# 生成并打开文档
doc-open:
    cargo doc --no-deps --open
