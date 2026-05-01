//! Detection 任务级 loss 周边工具。
//!
//! ## 与 `nn::nodes::raw_node::loss/` 的边界
//!
//! 本模块**不实现新的可微 loss 算子**——objectness / class 等通用 loss 节点都
//! 属于通用算子层（`src/nn/nodes/raw_node/loss/`），由 `Var` 用户 API 层
//! （`src/nn/var/ops/loss.rs`）通过 `mse_loss / bce_loss` 等链式方法暴露；
//! detection 专属的 IoU-family bbox loss 走拼接式实现，由本子模块的同侪
//! [`super::iou_loss`] 用基础算子 + autograd 拼出（不在 raw_node 节点层）。
//! 本文件只负责 detection 任务**特有的 loss 周边工具**：
//!
//! - 已实现：[`DetectionLossComponents`] / [`DetectionLossWeights`]——把已经
//!   完成正负样本匹配后的 bbox / objectness / class loss 按权重组合成单个总
//!   loss。**它本身没有新的反向逻辑**，只是 `Var` 算术运算的薄包装。
//! - 未来可能扩充（按需引入，不强求都是"多任务加权"形态）：assignment cost
//!   matrix（如 SimOTA / DSL）、quality / distribution target 构造（QFL / VFL
//!   / DFL）、heatmap target 构造（CenterNet）等。这些都属于 detection 任务级、
//!   不应污染通用算子层的 loss 周边工具。
//!
//! anchor matching、grid assignment、query matching 等检测器**特定**逻辑应留
//! 在 example 或下游 adapter 中（如 `vision::detection::adapter::yolo`）。

use crate::nn::{GraphError, Var};

/// detection 多任务 loss 权重。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionLossWeights {
    pub bbox: f32,
    pub objectness: f32,
    pub class: f32,
}

impl DetectionLossWeights {
    pub const fn new(bbox: f32, objectness: f32, class: f32) -> Self {
        Self {
            bbox,
            objectness,
            class,
        }
    }
}

impl Default for DetectionLossWeights {
    fn default() -> Self {
        Self {
            bbox: 1.0,
            objectness: 1.0,
            class: 1.0,
        }
    }
}

/// detection 多任务 loss 组件。
#[derive(Clone)]
pub struct DetectionLossComponents {
    pub bbox: Option<Var>,
    pub objectness: Option<Var>,
    pub class: Option<Var>,
}

impl DetectionLossComponents {
    pub fn new(bbox: Option<Var>, objectness: Option<Var>, class: Option<Var>) -> Self {
        Self {
            bbox,
            objectness,
            class,
        }
    }

    pub fn from_required(bbox: Var, objectness: Var, class: Var) -> Self {
        Self::new(Some(bbox), Some(objectness), Some(class))
    }

    /// 按权重组合存在的 loss 组件。
    ///
    /// 权重为 `0.0` 的组件会被跳过；至少需要一个存在且权重非零的组件。
    pub fn weighted_total(&self, weights: DetectionLossWeights) -> Result<Var, GraphError> {
        let mut total: Option<Var> = None;
        for (component, weight) in [
            (&self.bbox, weights.bbox),
            (&self.objectness, weights.objectness),
            (&self.class, weights.class),
        ] {
            let Some(component) = component else {
                continue;
            };
            if weight == 0.0 {
                continue;
            }
            let weighted = component * weight;
            total = Some(match total {
                Some(acc) => acc + weighted,
                None => weighted,
            });
        }
        total.ok_or_else(|| {
            GraphError::ComputationError(
                "detection loss 至少需要一个存在且权重非零的组件".to_string(),
            )
        })
    }
}
