/*
 * @Author       : 老董
 * @Date         : 2026-02-13
 * @Description  : 张量元素级操作：minimum、maximum、sign、abs、clip
 */

use super::super::next_source_id;
use crate::errors::TensorError;
use crate::tensor::Tensor;
use ndarray::{IxDyn, Zip};

impl Tensor {
    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓minimum/maximum↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 逐元素取两个张量的最小值
    ///
    /// 类似 `PyTorch` 的 `torch.minimum(a, b)` 或 `NumPy` 的 `np.minimum(a, b)`。
    /// 支持 NumPy 风格的广播（broadcasting）。
    ///
    /// # 参数
    /// - `other`: 另一个张量
    ///
    /// # 返回
    /// 新张量，形状为广播后的形状，每个元素为对应位置两个输入的较小值
    ///
    /// # Panics
    /// 如果两个张量的形状不兼容（无法广播）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 相同形状
    /// let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    /// let b = Tensor::new(&[2.0, 2.0, 5.0], &[3]);
    /// let result = a.minimum(&b);
    /// assert_eq!(result[[0]], 1.0);  // min(1, 2) = 1
    /// assert_eq!(result[[1]], 2.0);  // min(4, 2) = 2
    /// assert_eq!(result[[2]], 3.0);  // min(3, 5) = 3
    ///
    /// // 广播：标量与向量
    /// let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    /// let b = Tensor::new(&[2.0], &[1]);
    /// let result = a.minimum(&b);
    /// assert_eq!(result[[0]], 1.0);  // min(1, 2) = 1
    /// assert_eq!(result[[1]], 2.0);  // min(4, 2) = 2
    /// assert_eq!(result[[2]], 2.0);  // min(3, 2) = 2
    /// ```
    pub fn minimum(&self, other: &Tensor) -> Tensor {
        use crate::tensor::property::broadcast_shape;

        // 检查广播兼容性
        assert!(
            self.can_broadcast_with(other),
            "{}",
            TensorError::IncompatibleShape
        );

        // 计算广播后的形状
        let result_shape = broadcast_shape(self.shape(), other.shape()).expect("广播形状计算失败");

        // 使用 Zip 的 and_broadcast 实现逐元素最小值
        let result_data = Zip::from(self.data.broadcast(IxDyn(&result_shape)).unwrap())
            .and(other.data.broadcast(IxDyn(&result_shape)).unwrap())
            .map_collect(|&a, &b| a.min(b));

        Tensor {
            data: result_data,
            source_id: next_source_id(),
        }
    }

    /// 逐元素取两个张量的最大值
    ///
    /// 类似 `PyTorch` 的 `torch.maximum(a, b)` 或 `NumPy` 的 `np.maximum(a, b)`。
    /// 支持 NumPy 风格的广播（broadcasting）。
    ///
    /// # 参数
    /// - `other`: 另一个张量
    ///
    /// # 返回
    /// 新张量，形状为广播后的形状，每个元素为对应位置两个输入的较大值
    ///
    /// # Panics
    /// 如果两个张量的形状不兼容（无法广播）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // 相同形状
    /// let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    /// let b = Tensor::new(&[2.0, 2.0, 5.0], &[3]);
    /// let result = a.maximum(&b);
    /// assert_eq!(result[[0]], 2.0);  // max(1, 2) = 2
    /// assert_eq!(result[[1]], 4.0);  // max(4, 2) = 4
    /// assert_eq!(result[[2]], 5.0);  // max(3, 5) = 5
    ///
    /// // 广播：标量与向量
    /// let a = Tensor::new(&[1.0, 4.0, 3.0], &[3]);
    /// let b = Tensor::new(&[2.0], &[1]);
    /// let result = a.maximum(&b);
    /// assert_eq!(result[[0]], 2.0);  // max(1, 2) = 2
    /// assert_eq!(result[[1]], 4.0);  // max(4, 2) = 4
    /// assert_eq!(result[[2]], 3.0);  // max(3, 2) = 3
    /// ```
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        use crate::tensor::property::broadcast_shape;

        // 检查广播兼容性
        assert!(
            self.can_broadcast_with(other),
            "{}",
            TensorError::IncompatibleShape
        );

        // 计算广播后的形状
        let result_shape = broadcast_shape(self.shape(), other.shape()).expect("广播形状计算失败");

        // 使用 Zip 实现逐元素最大值
        let result_data = Zip::from(self.data.broadcast(IxDyn(&result_shape)).unwrap())
            .and(other.data.broadcast(IxDyn(&result_shape)).unwrap())
            .map_collect(|&a, &b| a.max(b));

        Tensor {
            data: result_data,
            source_id: next_source_id(),
        }
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑minimum/maximum↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓sign↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 返回张量每个元素的符号
    ///
    /// - 正数返回 1.0
    /// - 负数返回 -1.0
    /// - 零返回 0.0
    /// - NaN 返回 NaN
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    /// let y = x.sign();
    /// // y = [-1.0, -1.0, 0.0, 1.0, 1.0]
    /// ```
    pub fn sign(&self) -> Self {
        // 注：Rust 的 f32::signum() 对 0.0 返回 1.0，这与 PyTorch 行为不同
        // 这里显式处理零值，使其返回 0.0
        let data = self.data.mapv(|x| if x == 0.0 { 0.0 } else { x.signum() });
        Self {
            data,
            source_id: next_source_id(),
        }
    }

    /// 就地计算张量每个元素的符号
    pub fn sign_mut(&mut self) {
        self.data
            .mapv_inplace(|x| if x == 0.0 { 0.0 } else { x.signum() });
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑sign↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓abs↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 计算张量每个元素的绝对值
    ///
    /// - 正数返回原值
    /// - 负数返回其相反数
    /// - 零返回 0.0
    /// - NaN 返回 NaN
    /// - ±INFINITY 返回 INFINITY
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    /// let y = x.abs();
    /// // y = [2.0, 1.0, 0.0, 1.0, 2.0]
    /// ```
    pub fn abs(&self) -> Self {
        let data = self.data.mapv(f32::abs);
        Self {
            data,
            source_id: next_source_id(),
        }
    }

    /// 就地计算张量每个元素的绝对值
    pub fn abs_mut(&mut self) {
        self.data.mapv_inplace(f32::abs);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑abs↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓clip↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素进行值域裁剪: clip(x, min, max)
    ///
    /// 将每个元素限制在 `[min, max]` 范围内。
    /// 等价于 NumPy 的 `np.clip()` 或 PyTorch 的 `torch.clamp()`。
    ///
    /// # 参数
    /// - `min`: 下界
    /// - `max`: 上界（要求 `min <= max`）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[-3.0, -1.0, 0.0, 1.0, 3.0], &[5]);
    /// let y = x.clip(-2.0, 2.0);
    /// // y = [-2.0, -1.0, 0.0, 1.0, 2.0]
    /// ```
    pub fn clip(&self, min: f32, max: f32) -> Self {
        let data = self.data.mapv(|x| x.clamp(min, max));
        Self {
            data,
            source_id: next_source_id(),
        }
    }

    /// 就地对张量的每个元素进行值域裁剪
    pub fn clip_mut(&mut self, min: f32, max: f32) {
        self.data.mapv_inplace(|x| x.clamp(min, max));
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑clip↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
