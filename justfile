# Only-Torch Justfile
# 使用: just <command>

# ==================== BLAS 自动检测 ====================
# 优先级：MKL > OpenBLAS > 纯 Rust（matrixmultiply）
# 检测方式：环境变量 > 常见安装路径 > pkg-config/ldconfig
# 手动覆盖：设置 MKLROOT 或 OPENBLAS_DIR 环境变量即可

_detected_blas := ```
    if [ -n "$MKLROOT" ]; then
        echo "blas-mkl"
    elif [ -d "/c/Program Files (x86)/Intel/oneAPI/mkl/latest" ] || \
         [ -d "/c/Program Files/Intel/oneAPI/mkl/latest" ] || \
         [ -d "/opt/intel/oneapi/mkl/latest" ] || \
         [ -d "/opt/intel/mkl" ]; then
        echo "blas-mkl"
    elif [ -n "$OPENBLAS_DIR" ]; then
        echo "blas-openblas"
    elif pkg-config --exists openblas 2>/dev/null; then
        echo "blas-openblas"
    elif ldconfig -p 2>/dev/null | grep -q libopenblas; then
        echo "blas-openblas"
    fi
```

_blas_flag := if _detected_blas != "" { "--features " + _detected_blas } else { "" }
_blas_name := if _detected_blas == "blas-mkl" { "Intel MKL" } else if _detected_blas == "blas-openblas" { "OpenBLAS" } else { "pure Rust" }

# 默认命令：运行测试和 lint
default: test lint

# 完整检查：清理、测试、lint、所有 examples
all: clean test lint examples

# 查看 BLAS 检测状态
blas-status:
    @echo "BLAS backend: {{_blas_name}}"
    @echo "Cargo flag:   {{_blas_flag}}"

# ==================== 测试 ====================

# 运行常规单元测试（跳过 #[ignore] 标记的测试）
test:
    cargo test {{_blas_flag}}

# 运行全量测试（包含 #[ignore] 标记的联网等测试）
test-all:
    cargo test {{_blas_flag}} -- --include-ignored

# 仅运行 #[ignore] 标记的测试（联网等特殊测试）
test-ignored:
    cargo test {{_blas_flag}} -- --ignored

# 运行所有单元测试（单线程，用于调试）
test-serial:
    cargo test {{_blas_flag}} -- --test-threads=1

# 运行特定测试（用法: just test-filter <pattern>）
test-filter pattern:
    cargo test {{_blas_flag}} {{pattern}}

# ==================== 基准测试 ====================

# 运行所有 benchmarks
bench:
    cargo bench {{_blas_flag}}

# 运行特定 benchmark（用法: just bench-filter <pattern>）
bench-filter pattern:
    cargo bench {{_blas_flag}} -- {{pattern}}

# 单独 benchmark 组
bench-conv2d:
    cargo bench --bench conv2d {{_blas_flag}}

bench-backward:
    cargo bench --bench backward {{_blas_flag}}

bench-end-to-end:
    cargo bench --bench end_to_end {{_blas_flag}}

bench-tensor:
    cargo bench --bench tensor_ops {{_blas_flag}}

# ==================== Examples ====================

# 运行所有 examples（含 RL）
examples: example-xor example-iris example-sine example-mnist example-mnist-cnn example-mnist-gan example-california example-parity example-dual-input example-siamese example-dual-output example-multi-io example-multi-label example-cartpole-sac example-pendulum-sac example-moving-sac

# 运行所有 parity examples（RNN/LSTM/GRU）
example-parity: example-parity-fixed example-parity-var example-parity-lstm example-parity-gru

# 单个 examples
example-xor:
    @echo "=== Running XOR [{{_blas_name}}] ==="
    cargo run --example xor {{_blas_flag}}

example-iris:
    @echo "=== Running Iris [{{_blas_name}}] ==="
    cargo run --example iris {{_blas_flag}}

example-sine:
    @echo "=== Running Sine Regression [{{_blas_name}}] ==="
    cargo run --example sine_regression {{_blas_flag}}

example-mnist:
    @echo "=== Running MNIST [{{_blas_name}}] ==="
    cargo run --example mnist {{_blas_flag}}

example-mnist-cnn:
    @echo "=== Running MNIST CNN [{{_blas_name}}] ==="
    cargo run --example mnist_cnn {{_blas_flag}}

example-mnist-gan:
    @echo "=== Running MNIST GAN [{{_blas_name}}] ==="
    cargo run --example mnist_gan {{_blas_flag}}

example-california:
    @echo "=== Running California Housing [{{_blas_name}}] ==="
    cargo run --example california_housing {{_blas_flag}}

example-parity-fixed:
    @echo "=== Running Parity (RNN, Fixed Length) [{{_blas_name}}] ==="
    cargo run --example parity_rnn_fixed_len {{_blas_flag}}

example-parity-var:
    @echo "=== Running Parity (RNN, Variable Length) [{{_blas_name}}] ==="
    cargo run --example parity_rnn_var_len {{_blas_flag}}

example-parity-lstm:
    @echo "=== Running Parity (LSTM, Variable Length) [{{_blas_name}}] ==="
    cargo run --example parity_lstm_var_len {{_blas_flag}}

example-parity-gru:
    @echo "=== Running Parity (GRU, Variable Length) [{{_blas_name}}] ==="
    cargo run --example parity_gru_var_len {{_blas_flag}}

example-dual-input:
    @echo "=== Running Dual Input Add (forward2) [{{_blas_name}}] ==="
    cargo run --example dual_input_add {{_blas_flag}}

example-siamese:
    @echo "=== Running Siamese Similarity (shared encoder) [{{_blas_name}}] ==="
    cargo run --example siamese_similarity {{_blas_flag}}

example-dual-output:
    @echo "=== Running Dual Output Classify (multi-output) [{{_blas_name}}] ==="
    cargo run --example dual_output_classify {{_blas_flag}}

example-multi-io:
    @echo "=== Running Multi IO Fusion (multi-input + multi-output) [{{_blas_name}}] ==="
    cargo run --example multi_io_fusion {{_blas_flag}}

example-multi-label:
    @echo "=== Running Multi Label Point (BCE Loss) [{{_blas_name}}] ==="
    cargo run --example multi_label_point {{_blas_flag}}

example-cartpole-sac:
    @echo "=== Running CartPole SAC [{{_blas_name}}] (requires Python + gymnasium) ==="
    cargo run --example cartpole_sac {{_blas_flag}}

example-pendulum-sac:
    @echo "=== Running Pendulum SAC [{{_blas_name}}] (requires Python + gymnasium) ==="
    cargo run --example pendulum_sac {{_blas_flag}}

example-moving-sac:
    @echo "=== Running Moving Hybrid SAC [{{_blas_name}}] (requires Python + gymnasium) ==="
    cargo run --example moving_sac {{_blas_flag}}

# ==================== 代码质量 ====================

# 运行 clippy lint
lint:
    cargo clippy {{_blas_flag}}

# 运行 clippy 并自动修复
lint-fix:
    cargo clippy {{_blas_flag}} --fix --allow-dirty --allow-staged

# 格式化代码
fmt:
    cargo fmt

# 检查格式（不修改）
fmt-check:
    cargo fmt -- --check

# ==================== 构建 ====================

# Debug 构建
build:
    cargo build {{_blas_flag}}

# Release 构建
build-release:
    cargo build --release {{_blas_flag}}

# 检查编译（不生成二进制）
check:
    cargo check {{_blas_flag}}

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
