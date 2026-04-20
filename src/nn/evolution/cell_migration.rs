//! 循环单元类型切换（RNN ↔ LSTM ↔ GRU）的权重迁移。
//!
//! 和 Net2Net 不同，这里无法做到严格的函数保持（RNN/LSTM/GRU 的门/状态维数
//! 不同，信息量不对等），只能做 **informed initialization**：
//! 通过让新 cell 的无关门饱和为 0 或 1，使信号路径尽可能接近原 cell 的计算，
//! 为后续训练提供比纯随机初始化更好的起点。
//!
//! # 门顺序（与 `migration.rs::expand_*` 对齐）
//!
//! - RNN（3 参数）：`[W_ih, W_hh, b]` — 单门
//! - LSTM（12 参数，4 门，顺序 i/f/g/o）：
//!   `[W_ii, W_hi, b_i,  W_if, W_hf, b_f,  W_ig, W_hg, b_g,  W_io, W_ho, b_o]`
//! - GRU（9 参数，3 门，顺序 r/z/n）：
//!   `[W_ir, W_hr, b_r,  W_iz, W_hz, b_z,  W_in, W_hn, b_n]`
//!
//! # 迁移策略
//!
//! 用 `σ(±6) ≈ {1, 0}` 让饱和门行为确定：
//!
//! | 迁移        | 策略 |
//! |-------------|------|
//! | RNN  → LSTM | `g ← RNN`; `i, o` 饱和=1; `f` 饱和=0 → `c≈g, h≈tanh(c)` |
//! | RNN  → GRU  | `n ← RNN`; `r` 饱和=1; `z` 饱和=0 → `h≈n`               |
//! | LSTM → RNN  | 提取 `g` 门 (candidate)                                  |
//! | LSTM → GRU  | `n ← g`; `z ← f`; `r` 饱和=1                            |
//! | GRU  → RNN  | 提取 `n` 门 (candidate)                                  |
//! | GRU  → LSTM | `g ← n`; `f ← z`; `i, o` 饱和=1                         |
//!
//! 所有权重 `W_i*` 形状 `[in_dim, hidden]`，`W_h*` 形状 `[hidden, hidden]`，
//! `b_*` 形状 `[1, hidden]`。`MutateCellType` 保持 in_dim 和 hidden_size 不变，
//! 所以无需重新 reshape，只需"搬运+构造饱和"。

use std::collections::HashMap;

use crate::tensor::Tensor;

/// 让 sigmoid 饱和为 ≈1 的 bias 值（σ(6) ≈ 0.9975）。
const SIG_HI: f32 = 6.0;
/// 让 sigmoid 饱和为 ≈0 的 bias 值（σ(-6) ≈ 0.0025）。
const SIG_LO: f32 = -6.0;

/// 循环单元类型。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CellKind {
    Rnn,
    Lstm,
    Gru,
}

impl CellKind {
    /// 每种 cell 的参数张量数量。
    pub(crate) fn param_count(self) -> usize {
        match self {
            CellKind::Rnn => 3,
            CellKind::Lstm => 12,
            CellKind::Gru => 9,
        }
    }
}

/// 一个门的三元组：`(W_ih, W_hh, b)`。
#[derive(Clone)]
struct Gate {
    w_ih: Tensor,
    w_hh: Tensor,
    b: Tensor,
}

/// 根据现有 W_ih 形状构造一个"零权重 + 饱和 bias"的门。
///
/// 用于把某个门饱和成 sigmoid≈1 或 ≈0，从而在前向过程中让该门接近恒等/关闭。
fn saturated_gate(w_ih_shape: &[usize], hidden: usize, bias_value: f32) -> Gate {
    assert_eq!(w_ih_shape.len(), 2, "W_ih 期望二维");
    let in_dim = w_ih_shape[0];
    Gate {
        w_ih: Tensor::zeros(&[in_dim, hidden]),
        w_hh: Tensor::zeros(&[hidden, hidden]),
        b: Tensor::full(bias_value, &[1, hidden]),
    }
}

/// 从旧快照切片按"每 3 个"取出一个门。失败时返回 `None`（快照缺失等）。
fn gate_at(old_snaps: &[Option<Tensor>], gate_idx: usize) -> Option<Gate> {
    let base = gate_idx * 3;
    Some(Gate {
        w_ih: old_snaps.get(base)?.clone()?,
        w_hh: old_snaps.get(base + 1)?.clone()?,
        b: old_snaps.get(base + 2)?.clone()?,
    })
}

/// 迁移后每个门在新参数节点序列中的位置索引（基于门序）。
///
/// 例：LSTM 门序 [i, f, g, o]，返回 [gi, gf, gg, go] 表示顺序。
fn build_new_gates(
    old_kind: CellKind,
    new_kind: CellKind,
    old_snaps: &[Option<Tensor>],
    hidden: usize,
    shape_ref: &[usize],
) -> Option<Vec<Gate>> {
    match (old_kind, new_kind) {
        // --------- RNN → LSTM: [i, f, g, o] ---------
        (CellKind::Rnn, CellKind::Lstm) => {
            let rnn = gate_at(old_snaps, 0)?;
            Some(vec![
                saturated_gate(shape_ref, hidden, SIG_HI), // i ≈ 1
                saturated_gate(shape_ref, hidden, SIG_LO), // f ≈ 0
                rnn,                                       // g ← RNN
                saturated_gate(shape_ref, hidden, SIG_HI), // o ≈ 1
            ])
        }
        // --------- RNN → GRU: [r, z, n] ---------
        (CellKind::Rnn, CellKind::Gru) => {
            let rnn = gate_at(old_snaps, 0)?;
            Some(vec![
                saturated_gate(shape_ref, hidden, SIG_HI), // r ≈ 1
                saturated_gate(shape_ref, hidden, SIG_LO), // z ≈ 0
                rnn,                                       // n ← RNN
            ])
        }
        // --------- LSTM → RNN: 取 g 门 ---------
        (CellKind::Lstm, CellKind::Rnn) => {
            let g = gate_at(old_snaps, 2)?; // i=0, f=1, g=2, o=3
            Some(vec![g])
        }
        // --------- LSTM → GRU: [r, z, n] ---------
        (CellKind::Lstm, CellKind::Gru) => {
            let f = gate_at(old_snaps, 1)?;
            let g = gate_at(old_snaps, 2)?;
            Some(vec![
                saturated_gate(shape_ref, hidden, SIG_HI), // r ≈ 1
                f,                                         // z ← f
                g,                                         // n ← g
            ])
        }
        // --------- GRU → RNN: 取 n 门 ---------
        (CellKind::Gru, CellKind::Rnn) => {
            let n = gate_at(old_snaps, 2)?; // r=0, z=1, n=2
            Some(vec![n])
        }
        // --------- GRU → LSTM: [i, f, g, o] ---------
        (CellKind::Gru, CellKind::Lstm) => {
            let z = gate_at(old_snaps, 1)?;
            let n = gate_at(old_snaps, 2)?;
            Some(vec![
                saturated_gate(shape_ref, hidden, SIG_HI), // i ≈ 1
                z,                                         // f ← z
                n,                                         // g ← n
                saturated_gate(shape_ref, hidden, SIG_HI), // o ≈ 1
            ])
        }
        // Identity: 变异逻辑本就排除
        (a, b) if a == b => None,
        _ => None,
    }
}

/// 迁移主入口：生成 `new_param_ids` 对应的快照字典。
///
/// - `old_kind` / `new_kind`：旧/新 cell 类型。
/// - `old_snaps`：旧参数节点的快照，按 `expand_*` 的参数顺序排列（长度 = `old_kind.param_count()`）。
///   `None` 表示该参数无快照（被跳过，整个迁移将返回 `None`）。
/// - `new_param_ids`：新参数节点 id，按 `expand_*` 的顺序（长度 = `new_kind.param_count()`）。
/// - `hidden`：hidden_size（恒等迁移，不变）。
///
/// 成功返回 `Some(HashMap<new_param_id, Tensor>)`；任何信息缺失时返回 `None`，调用方
/// 应直接放弃写快照，让新参数走随机初始化。
pub(crate) fn migrate_cell_weights(
    old_kind: CellKind,
    old_snaps: &[Option<Tensor>],
    new_kind: CellKind,
    new_param_ids: &[u64],
    hidden: usize,
) -> Option<HashMap<u64, Tensor>> {
    if old_kind == new_kind {
        return None;
    }
    if old_snaps.len() != old_kind.param_count() {
        return None;
    }
    if new_param_ids.len() != new_kind.param_count() {
        return None;
    }

    // 用旧 W_ih 的形状作为新门 W_ih 的形状参考（in_dim 一致）
    let shape_ref = match old_snaps.first() {
        Some(Some(t)) => t.shape().to_vec(),
        _ => return None,
    };

    let gates = build_new_gates(old_kind, new_kind, old_snaps, hidden, &shape_ref)?;

    let mut out = HashMap::new();
    for (gate_idx, gate) in gates.into_iter().enumerate() {
        let base = gate_idx * 3;
        out.insert(new_param_ids[base], gate.w_ih);
        out.insert(new_param_ids[base + 1], gate.w_hh);
        out.insert(new_param_ids[base + 2], gate.b);
    }
    Some(out)
}
