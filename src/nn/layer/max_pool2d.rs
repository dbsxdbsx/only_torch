/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : MaxPool2d (2D 最大池化) 层 - PyTorch 风格 API
 *
 * 输入/输出形状：
 * - 输入：[batch_size, channels, H, W]
 * - 输出：[batch_size, channels, H', W']
 *
 * 输出尺寸计算：
 * H' = (H - kernel_h) / stride_h + 1
 * W' = (W - kernel_w) / stride_w + 1
 */

use crate::nn::graph::Graph;
use crate::nn::{IntoVar, Var};

// ==================== 新版 MaxPool2d 结构体（推荐）====================

/// `MaxPool2d` (2D 最大池化) 层
///
/// `PyTorch` 风格的最大池化层，无可学习参数。
///
/// # 输入/输出形状
/// - 输入：[`batch_size`, channels, H, W]
/// - 输出：[`batch_size`, channels, H', W']
///
/// # 输出尺寸计算
/// ```text
/// H' = (H - kernel_h) / stride_h + 1
/// W' = (W - kernel_w) / stride_w + 1
/// ```
///
/// # 使用示例
/// ```ignore
/// let pool = MaxPool2d::new(&graph, (2, 2), None, "pool1");
/// let h = pool.forward(&x);  // 下采样
/// ```
pub struct MaxPool2d {
    /// Graph 引用（用于 IntoVar 转换）
    graph: Graph,
    /// 池化窗口大小 (`kernel_h`, `kernel_w`)
    kernel_size: (usize, usize),
    /// 步长 (`stride_h`, `stride_w)，None` 时等于 `kernel_size`
    stride: Option<(usize, usize)>,
    /// 对称填充 (`pad_h`, `pad_w`)，等价于四角各填 (`pad_h`, `pad_h`, `pad_w`, `pad_w`)
    padding: (usize, usize),
    /// ONNX 风格 ceil_mode：true 用 ceil 计算输出尺寸
    ceil_mode: bool,
    /// 层名称（用于节点命名）
    name: String,
}

impl MaxPool2d {
    /// 创建新的 `MaxPool2d` 层（无 padding，PyTorch 默认行为）
    ///
    /// # 参数
    /// - `graph`: 计算图句柄
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，若为 None 则默认等于 `kernel_size`
    /// - `name`: 层名称前缀
    ///
    /// 需要 padding / ceil_mode 时用 [`MaxPool2d::with_padding`]
    pub fn new(
        graph: &Graph,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: &str,
    ) -> Self {
        Self {
            graph: graph.clone(),
            kernel_size,
            stride,
            padding: (0, 0),
            ceil_mode: false,
            name: name.to_string(),
        }
    }

    /// 创建带对称 padding / ceil_mode 的 `MaxPool2d` 层
    ///
    /// 主要用于 ONNX 导入路径（如 YOLOv5 SPPF 模块的 `MaxPool(k=5, pads=2, stride=1)`，
    /// 输出与输入同尺寸）。
    ///
    /// `padding` 是对称的 `(pad_h, pad_w)`,与 [`Conv2d`](super::Conv2d) 一致。
    pub fn with_padding(
        graph: &Graph,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
        ceil_mode: bool,
        name: &str,
    ) -> Self {
        Self {
            graph: graph.clone(),
            kernel_size,
            stride,
            padding,
            ceil_mode,
            name: name.to_string(),
        }
    }

    /// 前向传播
    ///
    /// # 参数
    /// - `x`: 输入，支持 `&Tensor`、`&Var` 等（自动转换）
    ///
    /// # 返回
    /// 输出 Var，形状 [`batch_size`, channels, H', W']
    pub fn forward(&self, x: impl IntoVar) -> Var {
        let x = x.into_var(&self.graph).expect("MaxPool2d 输入转换失败");
        let graph = x.get_graph();
        let node = graph
            .inner_mut()
            .create_max_pool2d_node(
                std::rc::Rc::clone(x.node()),
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                Some(&format!("{}_out", self.name)),
            )
            .expect("MaxPool2d forward 失败");
        Var::new_with_rc_graph(node, &graph.inner_rc())
    }

    /// 获取池化窗口大小
    pub const fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// 获取步长
    pub const fn stride(&self) -> Option<(usize, usize)> {
        self.stride
    }

    /// 获取对称 padding (`pad_h`, `pad_w`)
    pub const fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// 获取 ceil_mode
    pub const fn ceil_mode(&self) -> bool {
        self.ceil_mode
    }
}
