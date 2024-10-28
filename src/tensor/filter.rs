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
    /// let result = t.where_with_f32(
    ///     |x| x >= 0.0,
    ///     |_| 0.0,
    ///     |x| -x
    /// );
    /// ```
    pub fn where_with_f32<F, T, U>(&self, condition: F, true_fn: T, false_fn: U) -> Self
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

    /// 基于另一个张量的条件过滤函数，可以灵活处理张量中的元素
    ///
    /// # 参数
    /// * `other` - 用于比较的张量，其形状必须与当前张量相同
    /// * `condition` - 条件函数，接收当前张量和比较张量的对应元素值并返回bool
    /// * `true_fn` - 当条件为true时的值转换函数，可以使用两个张量的对应元素值
    /// * `false_fn` - 当条件为false时的值转换函数，可以使用两个张量的对应元素值
    ///
    /// # 示例
    /// ```
    /// use crate::tensor::Tensor;
    /// let t = Tensor::new(&[-1.0, 0.0, 1.0], &[3]);
    /// let y = Tensor::new(&[0.0, 0.0, 0.0], &[3]);
    ///
    /// // 类似 np.where(x >= y, x + y, x - y)
    /// let result = t.where_with_tensor(
    ///     &y,
    ///     |x, y| x >= y,
    ///     |x, y| x + y,  // 可以使用y的值
    ///     |x, y| x - y   // 可以使用y的值
    /// );
    /// ```
    pub fn where_with_tensor<F, T, U>(
        &self,
        other: &Self,
        condition: F,
        true_fn: T,
        false_fn: U,
    ) -> Self
    where
        F: Fn(f32, f32) -> bool,
        T: Fn(f32, f32) -> f32, // 修改为接收两个参数
        U: Fn(f32, f32) -> f32, // 修改为接收两个参数
    {
        assert!(
            self.is_same_shape(other),
            "两个张量的形状必须相同，当前张量形状为{:?}，比较张量形状为{:?}",
            self.shape(),
            other.shape()
        );

        let result = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| {
                if x.is_nan() || y.is_nan() {
                    f32::NAN
                } else if condition(x, y) {
                    true_fn(x, y) // 传入两个值
                } else {
                    false_fn(x, y) // 传入两个值
                }
            })
            .collect::<Vec<_>>();

        Self::new(&result, self.shape())
    }
}

/// 提供类似numpy的where语法糖，用于f32常量比较
///
/// # 示例
/// ```
/// use crate::tensor::Tensor;
/// let t = Tensor::new(&[-1.0, 0.0, 1.0], &[3]);
///
/// // 类似 np.where(x >= 0.0, x + 1.0, 0.5)
/// let result = tensor_where_f32!(x >= 0.0, x + 1.0, 0.5);
/// ```
#[macro_export]
macro_rules! tensor_where_f32 {
    // 基础模式匹配
    ($tensor:ident $op:tt $val:expr, $true_expr:expr, $false_expr:expr) => {{
        #[allow(unused)]
        $tensor.where_with_f32(
            |$tensor| $tensor $op $val,
            |$tensor| $true_expr,
            |$tensor| $false_expr
        )
    }};
}

/// 提供类似numpy的where语法糖，用于张量间比较
///
/// # 示例
/// ```
/// use crate::tensor::Tensor;
/// let t = Tensor::new(&[-1.0, 0.0, 1.0], &[3]);
/// let y = Tensor::new(&[0.0, 0.0, 0.0], &[3]);
///
/// // 类似 np.where(t >= y, t + y, t - y)
/// let result = tensor_where_tensor!(t >= y, t + y, t - y);
/// ```
#[macro_export]
macro_rules! tensor_where_tensor {
    // 基础匹配规则：处理简单表达式和复合表达式
    ($t:ident $op:tt $y:ident, $true_expr:expr, $false_expr:expr) => {{
        #[allow(unused)]
        $t.where_with_tensor(
            &$y,
            |$t, $y| $t $op $y,
            |$t, $y| $true_expr,
            |$t, $y| $false_expr
        )
    }};
}

// /// 提供统一的where语法糖，可以处理f32常量比较和张量间比较
// ///
// /// # 示例
// /// ```
// /// use crate::tensor::Tensor;
// /// let t = Tensor::new(&[-1.0, 0.0, 1.0], &[3]);
// /// let y = Tensor::new(&[0.0, 0.0, 0.0], &[3]);
// ///
// /// // f32常量比较
// /// let result = tensor_where!(t >= 0.0, t + 1.0, t - 1.0);
// ///
// /// // 张量间比较
// /// let result = tensor_where!(t >= y, t + y, t - y);
// /// ```
// #[macro_export]
// macro_rules! tensor_where {
//     // 1. 处理张量比较 - 当右操作数是标识符时
//     ($t:ident $op:tt $y:ident, $true_expr:expr, $false_expr:expr) => {{
//         tensor_where_tensor!($t $op $y, $true_expr, $false_expr)
//     }};

//     // 2. 处理f32常量比较 - 当右操作数是数值字面量或非标识符表达式时
//     ($t:ident $op:tt $val:literal, $true_expr:expr, $false_expr:expr) => {{
//         tensor_where_f32!($t $op $val, $true_expr, $false_expr)
//     }};

//     // 3. 处理f32变量比较 - 当右操作数是变量或表达式时
//     ($t:ident $op:tt $val:expr, $true_expr:expr, $false_expr:expr) => {{
//         let _val: f32 = $val; // 类型检查
//         tensor_where_f32!($t $op $val, $true_expr, $false_expr)
//     }};
// }
