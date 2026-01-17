/*
 * @Author       : 老董
 * @Date         : 2025-12-22
 * @Description  : AvgPool2d (2D 平均池化) 层 - PyTorch 风格 API
 *
 * 输入/输出形状：
 * - 输入：[batch_size, channels, H, W]
 * - 输出：[batch_size, channels, H', W']
 *
 * 输出尺寸计算：
 * H' = (H - kernel_h) / stride_h + 1
 * W' = (W - kernel_w) / stride_w + 1
 */

use crate::nn::Var;

// ==================== 新版 AvgPool2d 结构体（推荐）====================

/// AvgPool2d (2D 平均池化) 层
///
/// PyTorch 风格的平均池化层，无可学习参数。
///
/// # 输入/输出形状
/// - 输入：[batch_size, channels, H, W]
/// - 输出：[batch_size, channels, H', W']
///
/// # 输出尺寸计算
/// ```text
/// H' = (H - kernel_h) / stride_h + 1
/// W' = (W - kernel_w) / stride_w + 1
/// ```
///
/// # 使用示例
/// ```ignore
/// let pool = AvgPool2d::new((2, 2), None, "pool1");
/// let h = pool.forward(&x);  // 下采样
/// ```
pub struct AvgPool2d {
    /// 池化窗口大小 (kernel_h, kernel_w)
    kernel_size: (usize, usize),
    /// 步长 (stride_h, stride_w)，None 时等于 kernel_size
    stride: Option<(usize, usize)>,
    /// 层名称（用于节点命名）
    name: String,
}

impl AvgPool2d {
    /// 创建新的 AvgPool2d 层
    ///
    /// # 参数
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长 (sH, sW)，若为 None 则默认等于 kernel_size
    /// - `name`: 层名称前缀
    ///
    /// # 返回
    /// AvgPool2d 层实例
    pub fn new(kernel_size: (usize, usize), stride: Option<(usize, usize)>, name: &str) -> Self {
        Self {
            kernel_size,
            stride,
            name: name.to_string(),
        }
    }

    /// 前向传播
    ///
    /// # 参数
    /// - `x`: 输入 Var，形状 [batch_size, channels, H, W]
    ///
    /// # 返回
    /// 输出 Var，形状 [batch_size, channels, H', W']
    pub fn forward(&self, x: &Var) -> Var {
        let graph = x.get_graph();
        let mut g = graph.inner_mut();
        let out_id = g
            .new_avg_pool2d_node(
                x.node_id(),
                self.kernel_size,
                self.stride,
                Some(&format!("{}_out", self.name)),
            )
            .expect("AvgPool2d forward 失败");
        Var::new(out_id, graph.inner_rc())
    }

    /// 获取池化窗口大小
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// 获取步长
    pub fn stride(&self) -> Option<(usize, usize)> {
        self.stride
    }
}
