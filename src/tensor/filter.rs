use super::Tensor;

impl Tensor {
    /// 过滤张量中大于阈值的元素
    ///
    /// # 参数
    /// * `threshold` - 阈值
    /// * `true_value` - 当元素大于阈值时的替换值
    /// * `false_value` - 当元素不大于阈值时的替换值
    ///
    /// 注意：如果元素是 NaN，将保持 NaN
    pub fn where_greater_than(&self, threshold: f32, true_value: f32, false_value: f32) -> Self {
        let result = self
            .data
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    f32::NAN
                } else if x > threshold {
                    true_value
                } else {
                    false_value
                }
            })
            .collect::<Vec<_>>();

        Self::new(&result, self.shape())
    }

    /// 过滤张量中小于阈值的元素
    ///
    /// # 参数
    /// * `threshold` - 阈值
    /// * `true_value` - 当元素小于阈值时的替换值
    /// * `false_value` - 当元素不小于阈值时的替换值
    ///
    /// 注意：如果元素是 NaN，将保持 NaN
    pub fn where_lower_than(&self, threshold: f32, true_value: f32, false_value: f32) -> Self {
        let result = self
            .data
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    f32::NAN
                } else if x < threshold {
                    true_value
                } else {
                    false_value
                }
            })
            .collect::<Vec<_>>();

        Self::new(&result, self.shape())
    }

    /// 过滤张量中大于等于阈值的元素
    ///
    /// # 参数
    /// * `threshold` - 阈值
    /// * `true_value` - 当元素大于等于阈值时的替换值
    /// * `false_value` - 当元素小于阈值时的替换值
    ///
    /// 注意：如果元素是 NaN，将保持 NaN
    pub fn where_greater_equal_than(
        &self,
        threshold: f32,
        true_value: f32,
        false_value: f32,
    ) -> Self {
        let result = self
            .data
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    f32::NAN
                } else if x >= threshold {
                    true_value
                } else {
                    false_value
                }
            })
            .collect::<Vec<_>>();

        Self::new(&result, self.shape())
    }

    /// 过滤张量中小于等于阈值的元素
    ///
    /// # 参数
    /// * `threshold` - 阈值
    /// * `true_value` - 当元素小于等于阈值时的替换值
    /// * `false_value` - 当元素大于阈值时的替换值
    ///
    /// 注意：如果元素是 NaN，将保持 NaN
    pub fn where_lower_equal_than(
        &self,
        threshold: f32,
        true_value: f32,
        false_value: f32,
    ) -> Self {
        let result = self
            .data
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    f32::NAN
                } else if x <= threshold {
                    true_value
                } else {
                    false_value
                }
            })
            .collect::<Vec<_>>();

        Self::new(&result, self.shape())
    }

    /// 过滤张量中等于阈值的元素
    ///
    /// # 参数
    /// * `threshold` - 阈值
    /// * `true_value` - 当元素等于阈值时的替换值
    /// * `false_value` - 当元素不等于阈值时的替换值
    ///
    /// 注意：如果元素是 NaN，将保持 NaN
    pub fn where_equal(&self, threshold: f32, true_value: f32, false_value: f32) -> Self {
        let result = self
            .data
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    f32::NAN
                } else if (x - threshold).abs() < f32::EPSILON {
                    true_value
                } else {
                    false_value
                }
            })
            .collect::<Vec<_>>();

        Self::new(&result, self.shape())
    }

    /// 过滤张量中不等于阈值的元素
    ///
    /// # 参数
    /// * `threshold` - 阈值
    /// * `true_value` - 当元素不等于阈值时的替换值
    /// * `false_value` - 当元素等于阈值时的替换值
    ///
    /// 注意：如果元素是 NaN，将保持 NaN
    pub fn where_not_equal(&self, threshold: f32, true_value: f32, false_value: f32) -> Self {
        let result = self
            .data
            .iter()
            .map(|&x| {
                if x.is_nan() {
                    f32::NAN
                } else if (x - threshold).abs() >= f32::EPSILON {
                    true_value
                } else {
                    false_value
                }
            })
            .collect::<Vec<_>>();

        Self::new(&result, self.shape())
    }
}
