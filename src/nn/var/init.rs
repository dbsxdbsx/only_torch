use crate::tensor::Tensor;

// ==================== Init 枚举 ====================

/// 参数初始化策略
#[derive(Debug, Clone)]
pub enum Init {
    /// 常数初始化
    Constant(f32),
    /// 全零
    Zeros,
    /// 全一
    Ones,
    /// 正态分布（使用 Graph 的 RNG）
    Normal { mean: f32, std: f32 },
    /// Kaiming/He 初始化（适用于 `ReLU`）
    Kaiming,
    /// Xavier/Glorot 初始化（适用于 Sigmoid/Tanh）
    Xavier,
}

/// 计算 fan_in 和 fan_out（PyTorch 兼容）
///
/// 遵循 PyTorch `torch.nn.init._calculate_fan_in_and_fan_out` 的逻辑，
/// 但需适配本框架 Linear 层 weight layout `[in, out]`（PyTorch 为 `[out, in]`）。
///
/// - **Conv 权重 (≥3D)** `[C_out, C_in, kH, kW, ...]`:
///   - `fan_in  = C_in  × receptive_field_size`（即 `shape[1] × prod(shape[2..])`）
///   - `fan_out = C_out × receptive_field_size`（即 `shape[0] × prod(shape[2..])`）
///
/// - **Linear 权重 (2D)** `[in_features, out_features]`:
///   - `fan_in  = in_features`（`shape[0]`）
///   - `fan_out = out_features`（`shape[1]`）
///   - 注意：PyTorch Linear 的 layout 是 `[out, in]`，fan_in = `shape[1]`；
///     本框架 layout 是 `[in, out]`，fan_in = `shape[0]`，结果相同。
fn calculate_fan_in_and_fan_out(shape: &[usize]) -> (usize, usize) {
    assert!(
        shape.len() >= 2,
        "calculate_fan_in_and_fan_out: 至少需要 2 维，得到 {}D",
        shape.len()
    );

    if shape.len() == 2 {
        // Linear: [in_features, out_features]
        (shape[0], shape[1])
    } else {
        // Conv: [C_out, C_in, kH, kW, ...]
        let receptive_field_size: usize = shape[2..].iter().product();
        let fan_in = shape[1] * receptive_field_size;
        let fan_out = shape[0] * receptive_field_size;
        (fan_in, fan_out)
    }
}

impl Init {
    /// 生成初始化后的 Tensor（使用全局 RNG）
    pub fn generate(&self, shape: &[usize]) -> Tensor {
        match self {
            Self::Constant(v) => &Tensor::ones(shape) * *v,
            Self::Zeros => Tensor::zeros(shape),
            Self::Ones => Tensor::ones(shape),
            Self::Normal { mean, std } => Tensor::normal(*mean, *std, shape),
            Self::Kaiming => {
                let (fan_in, _) = calculate_fan_in_and_fan_out(shape);
                let std = (2.0 / fan_in as f32).sqrt();
                Tensor::normal(0.0, std, shape)
            }
            Self::Xavier => {
                let (fan_in, fan_out) = calculate_fan_in_and_fan_out(shape);
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::normal(0.0, std, shape)
            }
        }
    }

    /// 生成初始化后的 Tensor（使用指定的 RNG）
    pub fn generate_with_rng(&self, shape: &[usize], rng: &mut rand::rngs::StdRng) -> Tensor {
        match self {
            Self::Constant(v) => &Tensor::ones(shape) * *v,
            Self::Zeros => Tensor::zeros(shape),
            Self::Ones => Tensor::ones(shape),
            Self::Normal { mean, std } => Tensor::normal_with_rng(*mean, *std, shape, rng),
            Self::Kaiming => {
                let (fan_in, _) = calculate_fan_in_and_fan_out(shape);
                let std = (2.0 / fan_in as f32).sqrt();
                Tensor::normal_with_rng(0.0, std, shape, rng)
            }
            Self::Xavier => {
                let (fan_in, fan_out) = calculate_fan_in_and_fan_out(shape);
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::normal_with_rng(0.0, std, shape, rng)
            }
        }
    }
}
