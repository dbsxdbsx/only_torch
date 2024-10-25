use super::Tensor;

impl Tensor {
    /// 通用的条件过滤函数，可以灵活处理张量中的元素
    ///
    /// # 参数
    /// * `condition` - 条件函数，接收元素值并返回bool
    /// * `true_fn` - 当条件为true时的值转换函数
    /// * `false_fn` - 当条件为false时的值转换函数
    ///
    /// # 示例
    /// ```
    /// use crate::tensor::Tensor;
    /// let t = Tensor::new(&[-1.0, 0.0, 1.0], &[3]);
    ///
    /// // 类似 np.where(x >= 0.0, 0.0, -x)
    /// let result = t.where_with(
    ///     |x| x >= 0.0,
    ///     |_| 0.0,
    ///     |x| -x
    /// );
    /// ```
    pub fn where_with<F, T, U>(&self, condition: F, true_fn: T, false_fn: U) -> Self
    where
        F: Fn(f32) -> bool,
        T: Fn(f32) -> f32,
        U: Fn(f32) -> f32,
    {
        let result = self
            .data
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    f32::NAN
                } else if condition(x) {
                    true_fn(x)
                } else {
                    false_fn(x)
                }
            })
            .collect::<Vec<_>>();

        Self::new(&result, self.shape())
    }
}
