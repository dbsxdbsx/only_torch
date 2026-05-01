//! Detection 任务的类型契约。
//!
//! 这里只定义 backbone / head / assignment 的接口契约，让 only_torch 内不同
//! 检测器实现能够互换。**契约比实现先行**：本文件不提供任何具体 backbone /
//! head / assigner 实现，等第一个原生 detector（如 YOLO-lite）出现时再固定
//! trait 细节，避免反复回炉。
//!
//! 与本契约配套的"任务级"积木：
//! - `vision::detection::{BBox, Detection, GroundTruthBox}`：基础数据类型
//! - `vision::detection::nms`：通用后处理
//! - `vision::detection::loss`：通用 loss 组件加权（迁位中）
//! - `metrics::detection`：mAP / precision / recall

use crate::nn::{GraphError, Var};
use crate::tensor::Tensor;
use crate::vision::detection::{Detection, GroundTruthBox};

// ============================================================================
// Backbone
// ============================================================================

/// Backbone 的多尺度特征输出。
///
/// 检测器的 backbone 需要把图像编码为一个或多个尺度的特征图，并配套提供
/// stride 元数据，让后续 head / assigner 把特征坐标还原回图像坐标。
///
/// 字段约定：
/// - `features.len() == strides.len()`，按从浅到深排列。
/// - `features[i]` 形状必须是 `[N, C, H, W]`，且与原图满足
///   `H * strides[i] ≈ image_height`、`W * strides[i] ≈ image_width`（向下取整）。
/// - 单尺度 backbone 只返回 1 个元素的列表；YOLO 风格三尺度 backbone 通常
///   返回 strides `[8, 16, 32]`。
#[derive(Debug, Clone)]
pub struct BackboneOutput {
    pub features: Vec<Var>,
    pub strides: Vec<u32>,
}

impl BackboneOutput {
    /// 构造单尺度输出。
    pub fn single_scale(feature: Var, stride: u32) -> Self {
        Self {
            features: vec![feature],
            strides: vec![stride],
        }
    }

    /// 从 `(feature, stride)` 序列构造多尺度输出。
    pub fn multi_scale<I>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (Var, u32)>,
    {
        let (features, strides): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Self { features, strides }
    }

    /// 尺度数。
    pub fn num_scales(&self) -> usize {
        self.features.len()
    }

    /// 是否为空。
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// 通用 Backbone 契约。
///
/// 实现该 trait 的类型代表一个图像 backbone（可以是手写堆叠 Conv，也可以是
/// 经 ONNX 导入的预训练 backbone），返回多尺度特征图供 head / FPN / PAN 使用。
///
/// 注：本框架的 `Module` trait 只统一 `parameters()`，**不**统一 forward 签名
/// （详见 `.doc/design/node_vs_layer_design.md`）。`Backbone` trait 显式提供
/// detection 任务的 forward 契约，与 `Module` 正交——一个 backbone 类型既要
/// 实现 `Module`（汇报参数），也要实现 `Backbone`（约定 forward 形状）。
pub trait Backbone {
    /// 输入 `[N, C, H, W]` 图像 Var，输出多尺度特征。
    ///
    /// 通常 `C == 3`，但灰度 backbone 可以是 `C == 1`；具体由实现决定。
    fn forward(&self, image: &Var) -> Result<BackboneOutput, GraphError>;
}

// ============================================================================
// Detection Head 解码
// ============================================================================

/// Detection head 解码契约。
///
/// 把 head 的 raw 输出张量解码成结构化的 `Detection` 列表（按 batch 切分）。
/// 解码逻辑因检测器而异：YOLO 用 grid + anchor 解码；DETR 用 query 解码；
/// FCOS 用 anchor-free + center-ness 解码。**框架不假定特定方案**，留给具体
/// 实现自己定义。
///
/// 解码后通常会接 NMS（参考 `vision::detection::nms`）。
pub trait DetectionHeadDecode {
    /// 把 head 的 raw 输出按图像尺寸解码为 `Detection` 列表。
    ///
    /// 参数约定：
    /// - `raw`：head 直接输出的张量，形状由具体实现决定，第 0 维必须是 batch。
    /// - `image_size`：参考图像尺寸 `(width, height)`，单位像素。如果上游做了
    ///   letterbox，传 letterbox 输出尺寸；如果直接 resize，传 resize 后尺寸。
    ///   需要还原到原图坐标的，由调用者用 `LetterboxResult::bbox_to_origin`
    ///   等工具二次映射。
    ///
    /// 返回值：每张图一份 `Vec<Detection>`，外层长度等于 `raw` 的 batch 维。
    fn decode(
        &self,
        raw: &Tensor,
        image_size: (u32, u32),
    ) -> Result<Vec<Vec<Detection>>, GraphError>;
}

// ============================================================================
// Assignment / Matching
// ============================================================================

/// 单张图的 prediction → ground truth 匹配结果。
///
/// 字段约定：
/// - `pred_assignment[i]` 表示第 i 个 prediction 的归属：
///   - `Some(gt_idx)`：被分配给第 gt_idx 个 ground truth（正样本）。
///   - `None`：背景 / 负样本。
/// - `objectness_target` / `class_target`：可选辅助张量。具体 assigner 可按需
///   提供（如 IoU-aware assignment 会同时把 IoU 当作 quality 目标）。形状由
///   实现决定，使用方需要清楚自己的 assigner 输出什么。
#[derive(Debug, Clone)]
pub struct AssignmentResult {
    pub pred_assignment: Vec<Option<usize>>,
    pub objectness_target: Option<Tensor>,
    pub class_target: Option<Tensor>,
}

impl AssignmentResult {
    /// 仅按 prediction → GT 索引构造，不附带 objectness / class 目标。
    pub fn new(pred_assignment: Vec<Option<usize>>) -> Self {
        Self {
            pred_assignment,
            objectness_target: None,
            class_target: None,
        }
    }

    /// 附带 objectness 目标。
    pub fn with_objectness(mut self, objectness_target: Tensor) -> Self {
        self.objectness_target = Some(objectness_target);
        self
    }

    /// 附带 class 目标。
    pub fn with_class(mut self, class_target: Tensor) -> Self {
        self.class_target = Some(class_target);
        self
    }

    /// 正样本数。
    pub fn num_positives(&self) -> usize {
        self.pred_assignment
            .iter()
            .filter(|slot| slot.is_some())
            .count()
    }

    /// 负样本数。
    pub fn num_negatives(&self) -> usize {
        self.pred_assignment
            .iter()
            .filter(|slot| slot.is_none())
            .count()
    }
}

/// Assignment / matching 算法契约。
///
/// 把 head 的 predictions 与 ground truths 匹配。不同检测器有不同算法
/// （YOLO grid、ATSS、SimOTA、Hungarian / 匈牙利），但都可归结为
/// "给每个 prediction 决定它对应哪个 GT"。
///
/// `P` 是 prediction 的具体类型——故意做成泛型，让 assigner 实现自己决定
/// 怎么消费 head 的 raw 输出（可以是 `&Tensor`，也可以是某种已经预处理过
/// 的中间结构）。
pub trait Assigner<P> {
    /// 对单张图执行 assignment。
    ///
    /// - `predictions`：head 在该图上输出的 prediction 集合。
    /// - `ground_truths`：该图的 ground truth box 列表。
    /// - 返回值：每个 prediction 的归属信息。
    fn assign(
        &self,
        predictions: &P,
        ground_truths: &[GroundTruthBox],
    ) -> Result<AssignmentResult, GraphError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backbone_output_single_scale_metadata() {
        // 仅验证容器层面的元数据契约，不依赖任何具体 backbone 实现。
        let dummy = AssignmentResult::new(vec![Some(0), None, Some(1), None]);
        assert_eq!(dummy.num_positives(), 2);
        assert_eq!(dummy.num_negatives(), 2);

        let with_objectness =
            AssignmentResult::new(vec![Some(0)]).with_objectness(Tensor::new(&[1.0], &[1, 1]));
        assert!(with_objectness.objectness_target.is_some());
        assert_eq!(with_objectness.num_positives(), 1);
    }
}
