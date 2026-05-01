//! 通用 detection loss 组合工具。
//!
//! 本模块只负责把已完成匹配后的 bbox / objectness / class loss 按权重组合。
//! anchor matching、grid assignment、query matching 等检测器专属逻辑应留在
//! example 或下游 adapter 中。

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
