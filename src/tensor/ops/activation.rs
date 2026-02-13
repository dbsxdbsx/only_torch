/*
 * @Author       : 老董
 * @Date         : 2026-02-13
 * @Description  : 张量激活与数学函数：tanh、sigmoid、exp、ln、sqrt、softmax、log_softmax
 */

use super::super::next_source_id;
use crate::tensor::Tensor;

impl Tensor {
    /// 计算张量每个元素的平方根
    pub fn sqrt(&self) -> Self {
        let sqrt_data = self.data.mapv(f32::sqrt);
        Self { data: sqrt_data, source_id: next_source_id() }
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓tanh↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用双曲正切函数(tanh)
    ///
    /// tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    /// let y = x.tanh();
    /// // y ≈ [0.0, 0.7616, -0.7616]
    /// ```
    pub fn tanh(&self) -> Self {
        let data = self.data.mapv(f32::tanh);
        Self { data, source_id: next_source_id() }
    }

    /// 就地对张量的每个元素应用双曲正切函数(tanh)
    pub fn tanh_mut(&mut self) {
        self.data.mapv_inplace(f32::tanh);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑tanh↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓sigmoid↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素应用 Sigmoid 函数
    ///
    /// sigmoid(x) = 1 / (1 + e^(-x))
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, -1.0], &[3]);
    /// let y = x.sigmoid();
    /// // y ≈ [0.5, 0.7311, 0.2689]
    /// ```
    pub fn sigmoid(&self) -> Self {
        let data = self.data.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        Self { data, source_id: next_source_id() }
    }

    /// 就地对张量的每个元素应用 Sigmoid 函数
    pub fn sigmoid_mut(&mut self) {
        self.data.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑sigmoid↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓exp↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算指数函数 e^x
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[0.0, 1.0, 2.0], &[3]);
    /// let y = x.exp();
    /// // y ≈ [1.0, 2.7183, 7.3891]
    /// ```
    pub fn exp(&self) -> Self {
        let data = self.data.mapv(f32::exp);
        Self { data, source_id: next_source_id() }
    }

    /// 就地对张量的每个元素计算指数函数
    pub fn exp_mut(&mut self) {
        self.data.mapv_inplace(f32::exp);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑exp↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ln↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 对张量的每个元素计算自然对数 ln(x)
    ///
    /// # 注意
    /// 对于 x <= 0 的元素，结果为 NaN 或 -inf
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.7183, 7.3891], &[3]);
    /// let y = x.ln();
    /// // y ≈ [0.0, 1.0, 2.0]
    /// ```
    pub fn ln(&self) -> Self {
        let data = self.data.mapv(f32::ln);
        Self { data, source_id: next_source_id() }
    }

    /// 就地对张量的每个元素计算自然对数
    pub fn ln_mut(&mut self) {
        self.data.mapv_inplace(f32::ln);
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ln↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓softmax↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴计算 softmax（数值稳定版本）
    ///
    /// softmax(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
    ///
    /// 通过先减去最大值再计算 exp，避免数值溢出。
    ///
    /// # 参数
    /// - `axis`: 沿哪个轴计算 softmax
    ///
    /// # 返回
    /// 新张量，形状与输入相同，沿指定轴的元素和为 1
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let x = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    /// let probs = x.softmax(1);  // 沿最后一维计算
    ///
    /// // 每行和为 1
    /// assert!((probs[[0, 0]] + probs[[0, 1]] + probs[[0, 2]] - 1.0).abs() < 1e-6);
    /// assert!((probs[[1, 0]] + probs[[1, 1]] + probs[[1, 2]] - 1.0).abs() < 1e-6);
    ///
    /// // softmax([1,2,3]) ≈ [0.0900, 0.2447, 0.6652]
    /// assert!((probs[[0, 2]] - 0.6652).abs() < 0.001);
    /// ```
    pub fn softmax(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "softmax: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 数值稳定：先减去 max
        let max_vals = self.amax(axis); // 沿 axis 取最大值
        let max_broadcast = max_vals.unsqueeze(axis as i8); // 恢复维度以便广播

        // x - max(x)
        let shifted = self - &max_broadcast;

        // exp(x - max)
        let exp_vals = shifted.exp();

        // sum(exp)
        let sum_exp = exp_vals.sum_axis_keepdims(axis);

        // exp / sum
        &exp_vals / &sum_exp
    }

    /// 沿最后一维计算 softmax（数值稳定版本）
    ///
    /// 等价于 `self.softmax(self.dimension() - 1)`，这是最常用的情况。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    /// let probs = logits.softmax_last_dim();
    /// // probs ≈ [[0.0900, 0.2447, 0.6652]]
    /// ```
    pub fn softmax_last_dim(&self) -> Self {
        assert!(self.dimension() > 0, "softmax_last_dim: 张量维度必须大于 0");
        self.softmax(self.dimension() - 1)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑softmax↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓log_softmax↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 沿指定轴计算 log_softmax（数值稳定版本）
    ///
    /// `log_softmax(x) = log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))`
    ///
    /// 比直接计算 `softmax(x).ln()` 更数值稳定，避免 softmax 输出接近 0 时的精度问题。
    ///
    /// # 参数
    /// - `axis`: 计算 softmax 的轴
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
    /// let log_probs = logits.log_softmax(1);
    ///
    /// // 检查形状
    /// assert_eq!(log_probs.shape(), &[2, 3]);
    ///
    /// // log_softmax 输出应该都是负数（因为 softmax 输出 < 1）
    /// assert!(log_probs[[0, 0]] < 0.0);
    /// assert!(log_probs[[0, 1]] < 0.0);
    /// assert!(log_probs[[0, 2]] < 0.0);
    ///
    /// // exp(log_softmax) 应该等于 softmax
    /// let probs = log_probs.exp();
    /// let sum = probs[[0, 0]] + probs[[0, 1]] + probs[[0, 2]];
    /// assert!((sum - 1.0).abs() < 1e-6);
    /// ```
    pub fn log_softmax(&self, axis: usize) -> Self {
        assert!(
            axis < self.dimension(),
            "log_softmax: axis {} 超出维度范围 {}",
            axis,
            self.dimension()
        );

        // 数值稳定：先减去 max
        let max_vals = self.amax(axis);
        let max_broadcast = max_vals.unsqueeze(axis as i8);

        // shifted = x - max(x)
        let shifted = self - &max_broadcast;

        // log_sum_exp = log(sum(exp(shifted)))
        let exp_vals = shifted.exp();
        let sum_exp = exp_vals.sum_axis_keepdims(axis);
        let log_sum_exp = sum_exp.ln();

        // log_softmax = shifted - log_sum_exp
        &shifted - &log_sum_exp
    }

    /// 沿最后一维计算 log_softmax（数值稳定版本）
    ///
    /// 等价于 `self.log_softmax(self.dimension() - 1)`，这是最常用的情况。
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let logits = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    /// let log_probs = logits.log_softmax_last_dim();
    ///
    /// // log_softmax([1,2,3]) ≈ [-2.407, -1.407, -0.407]
    /// assert!((log_probs[[0, 0]] - (-2.407)).abs() < 0.01);
    /// ```
    pub fn log_softmax_last_dim(&self) -> Self {
        assert!(
            self.dimension() > 0,
            "log_softmax_last_dim: 张量维度必须大于 0"
        );
        self.log_softmax(self.dimension() - 1)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑log_softmax↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
