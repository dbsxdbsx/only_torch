/*
 * DynamicShape: 支持动态维度的形状系统
 *
 * 类似 Keras/TensorFlow 的 (None, 128) 设计，允许某些维度在编译时未知。
 * 例如 batch 维度通常是动态的，因为训练和推理时可能使用不同的 batch_size。
 *
 * # 示例
 * ```
 * use only_torch::nn::DynamicShape;
 *
 * // 固定形状
 * let fixed = DynamicShape::fixed(&[32, 128]);
 * assert_eq!(fixed.to_string(), "[32, 128]");
 *
 * // 动态 batch
 * let dynamic_batch = DynamicShape::with_dynamic_batch(&[128]);
 * assert_eq!(dynamic_batch.to_string(), "[?, 128]");
 *
 * // 完全自定义
 * let custom = DynamicShape::new(&[None, Some(10), None, Some(64)]);
 * assert_eq!(custom.to_string(), "[?, 10, ?, 64]");
 * ```
 */

use std::fmt;

/// 维度值：Some(n) 表示固定值 n，None 表示动态（任意值）
pub type Dim = Option<usize>;

/// 动态形状：支持动态维度的形状表示
///
/// 与 `Vec<usize>` 的区别：
/// - `Vec<usize>`: 所有维度必须是确定的数值
/// - `DynamicShape`: 某些维度可以是 None，表示"任意值"
///
/// # 使用场景
/// - batch 维度：训练时 256，推理时 1，用 None 表示
/// - 序列长度：变长序列，用 None 表示
/// - 可视化：None 显示为 `?`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicShape {
    dims: Vec<Dim>,
}

impl DynamicShape {
    /// 创建一个动态形状
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let shape = DynamicShape::new(&[None, Some(128)]);
    /// assert_eq!(shape.ndim(), 2);
    /// assert!(shape.is_dynamic(0));
    /// assert!(!shape.is_dynamic(1));
    /// ```
    pub fn new(dims: &[Dim]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }

    /// 从固定形状创建（所有维度都是确定的）
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let shape = DynamicShape::fixed(&[32, 128]);
    /// assert!(!shape.has_dynamic_dims());
    /// ```
    pub fn fixed(dims: &[usize]) -> Self {
        Self {
            dims: dims.iter().map(|&d| Some(d)).collect(),
        }
    }

    /// 创建一个动态 batch 的形状
    ///
    /// 第一维是 None（动态），其余维度固定。
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let shape = DynamicShape::with_dynamic_batch(&[128, 64]);
    /// assert_eq!(shape.to_string(), "[?, 128, 64]");
    /// ```
    pub fn with_dynamic_batch(feature_dims: &[usize]) -> Self {
        let mut dims = vec![None];
        dims.extend(feature_dims.iter().map(|&d| Some(d)));
        Self { dims }
    }

    /// 从实际张量形状创建固定形状
    pub fn from_tensor_shape(shape: &[usize]) -> Self {
        Self::fixed(shape)
    }

    /// 获取维度数量
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// 获取指定维度的值
    ///
    /// 返回 Some(n) 如果维度固定，None 如果维度动态
    pub fn dim(&self, index: usize) -> Dim {
        self.dims.get(index).copied().flatten()
    }

    /// 检查指定维度是否是动态的
    pub fn is_dynamic(&self, index: usize) -> bool {
        self.dims.get(index).map(|d| d.is_none()).unwrap_or(false)
    }

    /// 检查是否有任何动态维度
    pub fn has_dynamic_dims(&self) -> bool {
        self.dims.iter().any(|d| d.is_none())
    }

    /// 获取特征形状（忽略第一维 batch）
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let shape = DynamicShape::fixed(&[32, 128, 64]);
    /// let features = shape.feature_shape();
    /// assert_eq!(features.to_vec_fixed().unwrap(), vec![128, 64]);
    /// ```
    pub fn feature_shape(&self) -> DynamicShape {
        if self.dims.len() > 1 {
            DynamicShape::new(&self.dims[1..])
        } else {
            self.clone()
        }
    }

    /// 将第一维设置为动态
    pub fn with_batch_dynamic(&self) -> DynamicShape {
        if self.dims.is_empty() {
            return self.clone();
        }
        let mut dims = self.dims.clone();
        dims[0] = None;
        DynamicShape::new(&dims)
    }

    /// 检查此形状是否与另一个形状兼容
    ///
    /// 兼容规则：
    /// - 维度数量必须相同
    /// - 对于每个维度：两者都是 None，或者至少有一个是 None，或者值相等
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let a = DynamicShape::new(&[None, Some(128)]);
    /// let b = DynamicShape::fixed(&[32, 128]);
    /// let c = DynamicShape::fixed(&[16, 128]);
    /// let d = DynamicShape::fixed(&[32, 64]);
    ///
    /// assert!(a.is_compatible(&b));  // [?, 128] vs [32, 128] ✓
    /// assert!(a.is_compatible(&c));  // [?, 128] vs [16, 128] ✓
    /// assert!(!a.is_compatible(&d)); // [?, 128] vs [32, 64] ✗ (128 != 64)
    /// ```
    pub fn is_compatible(&self, other: &DynamicShape) -> bool {
        if self.dims.len() != other.dims.len() {
            return false;
        }
        self.dims
            .iter()
            .zip(other.dims.iter())
            .all(|(a, b)| match (a, b) {
                (None, _) | (_, None) => true,
                (Some(x), Some(y)) => x == y,
            })
    }

    /// 检查此形状是否与实际张量形状兼容
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let shape = DynamicShape::new(&[None, Some(128)]);
    /// assert!(shape.is_compatible_with_tensor(&[32, 128]));
    /// assert!(shape.is_compatible_with_tensor(&[1, 128]));
    /// assert!(!shape.is_compatible_with_tensor(&[32, 64]));
    /// assert!(!shape.is_compatible_with_tensor(&[32, 128, 10])); // 维度数不匹配
    /// ```
    pub fn is_compatible_with_tensor(&self, tensor_shape: &[usize]) -> bool {
        if self.dims.len() != tensor_shape.len() {
            return false;
        }
        self.dims
            .iter()
            .zip(tensor_shape.iter())
            .all(|(expected, &actual)| match expected {
                None => true,
                Some(n) => *n == actual,
            })
    }

    /// 合并两个形状，取更具体的值
    ///
    /// 如果两个形状不兼容，返回 None。
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let a = DynamicShape::new(&[None, Some(128)]);
    /// let b = DynamicShape::fixed(&[32, 128]);
    /// let merged = a.merge(&b).unwrap();
    /// assert_eq!(merged.to_string(), "[32, 128]");
    /// ```
    pub fn merge(&self, other: &DynamicShape) -> Option<DynamicShape> {
        if self.dims.len() != other.dims.len() {
            return None;
        }

        let merged: Option<Vec<Dim>> = self
            .dims
            .iter()
            .zip(other.dims.iter())
            .map(|(a, b)| match (a, b) {
                (None, None) => Some(None),
                (Some(x), None) | (None, Some(x)) => Some(Some(*x)),
                (Some(x), Some(y)) => {
                    if x == y {
                        Some(Some(*x))
                    } else {
                        None
                    }
                }
            })
            .collect();

        merged.map(|dims| DynamicShape { dims })
    }

    /// 计算两个形状广播后的动态形状
    ///
    /// 用于 Add、Multiply 等需要广播的操作。
    /// 如果任一维度是动态的，输出该维度也是动态的。
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let a = DynamicShape::new(&[None, Some(128)]);
    /// let b = DynamicShape::fixed(&[1, 128]);
    /// let result = a.broadcast_with(&b);
    /// assert_eq!(result.to_string(), "[?, 128]");
    /// ```
    pub fn broadcast_with(&self, other: &DynamicShape) -> DynamicShape {
        let max_ndim = self.dims.len().max(other.dims.len());

        // 从右边对齐（广播规则）
        let self_padded: Vec<Dim> = std::iter::repeat(&None)
            .take(max_ndim - self.dims.len())
            .cloned()
            .chain(self.dims.iter().cloned())
            .collect();

        let other_padded: Vec<Dim> = std::iter::repeat(&None)
            .take(max_ndim - other.dims.len())
            .cloned()
            .chain(other.dims.iter().cloned())
            .collect();

        let result: Vec<Dim> = self_padded
            .iter()
            .zip(other_padded.iter())
            .map(|(a, b)| match (a, b) {
                // 任一是动态的，结果也是动态的
                (None, _) | (_, None) => None,
                // 两个都是固定的，取较大的（广播规则）
                (Some(x), Some(y)) => Some(*x.max(y)),
            })
            .collect();

        DynamicShape { dims: result }
    }

    /// 使用实际张量形状具体化动态维度
    ///
    /// 返回一个完全固定的形状。
    ///
    /// # 示例
    /// ```
    /// use only_torch::nn::DynamicShape;
    ///
    /// let shape = DynamicShape::new(&[None, Some(128)]);
    /// let concrete = shape.concretize(&[32, 128]).unwrap();
    /// assert_eq!(concrete.to_vec_fixed().unwrap(), vec![32, 128]);
    /// ```
    pub fn concretize(&self, tensor_shape: &[usize]) -> Option<DynamicShape> {
        if !self.is_compatible_with_tensor(tensor_shape) {
            return None;
        }
        Some(DynamicShape::fixed(tensor_shape))
    }

    /// 转换为固定形状向量（如果所有维度都是固定的）
    ///
    /// 如果有任何动态维度，返回 None。
    pub fn to_vec_fixed(&self) -> Option<Vec<usize>> {
        self.dims.iter().map(|&d| d).collect()
    }

    /// 获取内部维度数组的引用
    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    /// 转换为用于显示的字符串（动态维度显示为 ?）
    pub fn to_display_string(&self) -> String {
        let parts: Vec<String> = self
            .dims
            .iter()
            .map(|d| match d {
                Some(n) => n.to_string(),
                None => "?".to_string(),
            })
            .collect();
        format!("[{}]", parts.join(", "))
    }
}

impl fmt::Display for DynamicShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_display_string())
    }
}

/// 从固定形状转换
impl From<&[usize]> for DynamicShape {
    fn from(shape: &[usize]) -> Self {
        DynamicShape::fixed(shape)
    }
}

impl From<Vec<usize>> for DynamicShape {
    fn from(shape: Vec<usize>) -> Self {
        DynamicShape::fixed(&shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_shape_creation() {
        // 固定形状
        let fixed = DynamicShape::fixed(&[32, 128]);
        assert_eq!(fixed.ndim(), 2);
        assert!(!fixed.has_dynamic_dims());
        assert_eq!(fixed.dim(0), Some(32));
        assert_eq!(fixed.dim(1), Some(128));

        // 动态 batch
        let dynamic_batch = DynamicShape::with_dynamic_batch(&[128]);
        assert_eq!(dynamic_batch.ndim(), 2);
        assert!(dynamic_batch.has_dynamic_dims());
        assert!(dynamic_batch.is_dynamic(0));
        assert!(!dynamic_batch.is_dynamic(1));
        assert_eq!(dynamic_batch.dim(0), None);
        assert_eq!(dynamic_batch.dim(1), Some(128));

        // 自定义
        let custom = DynamicShape::new(&[None, Some(10), None, Some(64)]);
        assert_eq!(custom.ndim(), 4);
        assert!(custom.is_dynamic(0));
        assert!(!custom.is_dynamic(1));
        assert!(custom.is_dynamic(2));
        assert!(!custom.is_dynamic(3));
    }

    #[test]
    fn test_dynamic_shape_display() {
        assert_eq!(DynamicShape::fixed(&[32, 128]).to_string(), "[32, 128]");
        assert_eq!(
            DynamicShape::with_dynamic_batch(&[128]).to_string(),
            "[?, 128]"
        );
        assert_eq!(
            DynamicShape::new(&[None, Some(10), None]).to_string(),
            "[?, 10, ?]"
        );
    }

    #[test]
    fn test_dynamic_shape_compatibility() {
        let dynamic = DynamicShape::new(&[None, Some(128)]);
        let fixed1 = DynamicShape::fixed(&[32, 128]);
        let fixed2 = DynamicShape::fixed(&[16, 128]);
        let fixed3 = DynamicShape::fixed(&[32, 64]);
        let fixed4 = DynamicShape::fixed(&[32, 128, 10]);

        // 动态与固定兼容
        assert!(dynamic.is_compatible(&fixed1));
        assert!(dynamic.is_compatible(&fixed2));

        // 固定维度不匹配
        assert!(!dynamic.is_compatible(&fixed3));

        // 维度数不同
        assert!(!dynamic.is_compatible(&fixed4));

        // 与张量形状兼容
        assert!(dynamic.is_compatible_with_tensor(&[32, 128]));
        assert!(dynamic.is_compatible_with_tensor(&[1, 128]));
        assert!(!dynamic.is_compatible_with_tensor(&[32, 64]));
    }

    #[test]
    fn test_dynamic_shape_merge() {
        let a = DynamicShape::new(&[None, Some(128)]);
        let b = DynamicShape::fixed(&[32, 128]);

        let merged = a.merge(&b).unwrap();
        assert_eq!(merged.to_string(), "[32, 128]");

        // 不兼容的形状无法合并
        let c = DynamicShape::fixed(&[32, 64]);
        assert!(a.merge(&c).is_none());

        // 两个动态合并仍是动态
        let d = DynamicShape::new(&[None, Some(128)]);
        let merged2 = a.merge(&d).unwrap();
        assert_eq!(merged2.to_string(), "[?, 128]");
    }

    #[test]
    fn test_dynamic_shape_concretize() {
        let shape = DynamicShape::new(&[None, Some(128)]);

        let concrete = shape.concretize(&[32, 128]).unwrap();
        assert_eq!(concrete.to_vec_fixed().unwrap(), vec![32, 128]);

        // 不兼容的形状无法具体化
        assert!(shape.concretize(&[32, 64]).is_none());
    }

    #[test]
    fn test_feature_shape() {
        let shape = DynamicShape::fixed(&[32, 128, 64]);
        let features = shape.feature_shape();
        assert_eq!(features.to_string(), "[128, 64]");

        let dynamic = DynamicShape::with_dynamic_batch(&[128, 64]);
        let features2 = dynamic.feature_shape();
        assert_eq!(features2.to_string(), "[128, 64]");
    }

    #[test]
    fn test_with_batch_dynamic() {
        let fixed = DynamicShape::fixed(&[32, 128]);
        let dynamic = fixed.with_batch_dynamic();
        assert_eq!(dynamic.to_string(), "[?, 128]");
    }

    #[test]
    fn test_to_vec_fixed() {
        let fixed = DynamicShape::fixed(&[32, 128]);
        assert_eq!(fixed.to_vec_fixed(), Some(vec![32, 128]));

        let dynamic = DynamicShape::with_dynamic_batch(&[128]);
        assert_eq!(dynamic.to_vec_fixed(), None);
    }

    #[test]
    fn test_broadcast_with() {
        // 动态与固定广播
        let a = DynamicShape::new(&[None, Some(128)]);
        let b = DynamicShape::fixed(&[1, 128]);
        let result = a.broadcast_with(&b);
        assert_eq!(result.to_string(), "[?, 128]");

        // 不同维度数的广播（右对齐）
        let c = DynamicShape::fixed(&[128]);
        let d = DynamicShape::fixed(&[32, 128]);
        let result2 = c.broadcast_with(&d);
        // 右对齐后 [?, 128] 和 [32, 128]
        // 左边扩展用 None 填充，结果是 [?, 128]
        assert_eq!(result2.ndim(), 2);

        // 两个固定形状广播
        let e = DynamicShape::fixed(&[1, 128]);
        let f = DynamicShape::fixed(&[32, 128]);
        let result3 = e.broadcast_with(&f);
        assert_eq!(result3.to_string(), "[32, 128]");

        // 动态维度传播
        let g = DynamicShape::new(&[None, Some(64)]);
        let h = DynamicShape::new(&[Some(32), None]);
        let result4 = g.broadcast_with(&h);
        // 任一动态则结果动态
        assert_eq!(result4.to_string(), "[?, ?]");
    }
}
