# AGENTS.md

尽量用中文说话。当然，相关的术语你可以用英文或者一些原文字。
本项目相关注释也尽量用中文

## Project Overview

**only_torch** is a toy-level PyTorch-like AI framework written in pure Rust (no C++, no Python bindings). It features:
- Dynamic computation graph with autograd (PyTorch-style `Graph` + `Var` + `Module` + `Optimizer`)
- CPU-only by design (no GPU support, targeting cross-platform including Android)
- NEAT-style neural architecture evolution (auto-discover network structure from minimal topology)
- All data types are f32; the core tensor type wraps `ndarray::ArrayD<f32>`
- Optional BLAS acceleration via feature flags (`blas-mkl`, `blas-openblas`)
- Reinforcement learning bridge to Python Gymnasium via `pyo3`

Primary language is Chinese (comments, docs, commit messages, error messages).

## Build & Development Commands

This project uses `just` (justfile) as task runner with automatic BLAS backend detection.

```
just build            # Debug build (auto-detects BLAS)
just build-release    # Release build with LTO
just check            # Compile check only (no binary)
just test             # Unit tests (skips #[ignore])
just test-all         # All tests including #[ignore] (network-dependent)
just test-filter <p>  # Run tests matching pattern
just test-serial      # Single-threaded tests (for debugging)
just lint             # cargo clippy
just lint-fix         # clippy --fix
just fmt              # cargo fmt
just fmt-check        # Format check
just bench            # All benchmarks (criterion)
just bench-conv2d     # Conv2d benchmark only
just doc              # Generate docs
just doc-open         # Generate and open docs
```

Running a single example:
```
cargo run --example xor
just example-xor
```

Running a specific test by name:
```
cargo test test_name
just test-filter test_name
```

RL examples require Python + gymnasium:
```
just example-cartpole-sac
```

## Architecture

### Core Abstraction Layers

The framework follows a layered architecture mirroring PyTorch:

1. **`tensor/`** — `Tensor` struct wrapping `ndarray::ArrayD<f32>`. Pure data, no graph awareness. Implements arithmetic, shape ops, reductions, activations, serialization. All operations are value-level (no autograd here).

2. **`nn/nodes/`** — Computation graph nodes. Each op (add, matmul, relu, conv2d, losses, etc.) is a `RawNode` variant in `nn/nodes/raw_node/`. Nodes store forward logic and `backward` (Jacobian) logic. New ops are added here.

3. **`nn/var/`** — `Var` is the user-facing smart handle (like PyTorch's `Tensor` with grad). It holds `Rc<NodeInner>` + `Weak<RefCell<GraphInner>>`. Operator overloading (`+`, `-`, `*`, `/`) and chainable methods (`.relu()`, `.matmul()`, `.cross_entropy()`) are defined on `Var`. Op trait groups: `VarActivationOps`, `VarLossOps`, `VarShapeOps`, `VarMatrixOps`, `VarReduceOps`, `VarSelectionOps`, `VarFilterOps`, `VarRegularizationOps`.

4. **`nn/graph/`** — `Graph` (user handle) wraps `Rc<RefCell<GraphInner>>`. `GraphInner` owns the computation graph state, forward/backward execution, parameter registry, mode (train/eval), visualization, and serialization. `Graph` is Clone (shared ownership).

5. **`nn/layer/`** — Higher-level modules: `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Rnn`, `Lstm`, `Gru`, `BatchNorm`, `LayerNorm`, `Embedding`, `MultiHeadAttention`, etc. Each implements the `Module` trait (`parameters() -> Vec<Var>`). Note: `forward()` is NOT a trait method (signatures vary per layer).

6. **`nn/optimizer/`** — `SGD` and `Adam`, implementing the `Optimizer` trait.

7. **`nn/distributions/`** — `Categorical`, `Normal`, `TanhNormal` for RL policy distributions.

8. **`nn/evolution/`** — NEAT-style neural architecture evolution. Key flow: `Evolution::supervised(...)` → genome-centric loop (build → restore weights → train → capture weights → evaluate → accept/rollback → mutate). Core files: `gene.rs` (genome data), `mutation.rs` (mutation ops), `builder.rs` (genome→Graph), `convergence.rs` (convergence detection), `task.rs` (task trait).

### Supporting Modules

- **`data/`** — `DataLoader`, `TensorDataset`, `VarLenDataset`, `BucketedDataLoader`; built-in datasets (`MnistDataset`, `CaliforniaHousingDataset`); data transforms (normalize, augmentation, crop, flip, etc.).
- **`rl/`** — `GymEnv` (Rust↔Python Gymnasium bridge via pyo3), `Step`, `MinariDataset`. RL examples use `pyo3` with `serial_test` in tests (Python module import races).
- **`metrics/`** — Classification metrics (accuracy, precision, recall, F1, confusion matrix, multi-label) and regression metrics (MSE, MAE, R²).
- **`vision/`** — Basic image I/O (`Vision::load_image`, `Vision::save_image`), grayscale conversion.
- **`errors/`** — `TensorError` and op-level error types.

### Key Design Decisions

- **Explicit broadcast**: Unlike PyTorch's implicit broadcast, this framework uses explicit broadcast nodes in the graph. This is intentional for NEAT evolution and cleaner gradient computation (see `.doc/design/broadcast_mechanism_design.md`).
- **Gradient flow control**: `Var::detach()` creates a gradient-blocking node (for GAN training, Actor-Critic, etc.). `graph.no_grad(|| {...})` disables gradient tracking. See `.doc/design/gradient_flow_control_design.md`.
- **Manual gradient clearing**: Users must call `optimizer.zero_grad()` before each backward pass (same as PyTorch). See `.doc/design/gradient_clear_and_accumulation_design.md`.
- **Module trait**: Only `parameters()` is a trait method. `forward()` and `new()` have varying signatures per layer and are NOT trait methods.
- **Node group context**: RAII guards (`NodeGroupContext`) automatically tag nodes for visualization clustering (layers, distributions).

### Test Organization

- Unit tests live as `mod tests` inside each source module (e.g., `src/nn/evolution/tests/`)
- Integration tests: `tests/` (top-level `.rs` files). Each integration test file should contain only one `#[test]`. If the test involves network topology, output visualization results.
- Python reference tests: `tests/*.py` (PyTorch reference implementations for numerical verification). For complex computations, first compute reference values in Python (PyTorch/JAX/Keras), then use those as expected values in Rust tests.
- Python gym tests: `tests/python/gym/`
- Archive tests: `tests/archive/` (early integration tests)
- pyo3 tests use `serial_test` crate to avoid Python import races
- Prefer debug builds for testing; use release only when specifically needed.
- Use `assert_panic!`, `assert_err!()` and similar macros to keep test code concise.

### Benchmarks

Four criterion benchmark suites in `benches/`: `conv2d`, `backward`, `end_to_end`, `tensor_ops`.

### Feature Flags

- `blas-mkl` — Intel MKL acceleration (auto-downloads if not installed)
- `blas-openblas` — OpenBLAS acceleration
- Default: pure Rust backend (no external dependencies)

### Internal Reference

- **MatrixSlow** (`./MatrixSlow`): Python deep learning framework bundled in-repo. Primary design reference — consult when facing architectural bottlenecks.
- **Design docs** in `.doc/design/` cover broadcast mechanism, gradient design, batch mechanism, DataLoader, serialization, visualization, memory/recurrence, evolution, distributions, RL roadmap, and future enhancements. Consult these before making architectural changes.
- **Terminology**: see `.doc/terminology_convention.md` for project-specific term definitions.

## Coding Conventions

- **Language**: all comments, docs, and commit messages in Chinese; English for technical terms only.
- **Privacy**: never expose local filesystem paths or private content in public-facing text (README, code comments, commit messages, this file).
- **Architecture perspective**: approach from a senior ML architect's viewpoint — ensure all features are CPU-efficient, ergonomic, extensible, and easy to integrate.
