use thiserror::Error;
mod ops;
pub use self::ops::*;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum TensorError {
    // 数字比较用
    #[error("{value_name}须{operator}{threshold}")]
    ValueMustSatisfyComparison {
        value_name: String,
        operator: ComparisonOperator,
        threshold: usize,
    },
    // 张量二元运算
    #[error(
        "形状不一致，故无法{operator}：第一个张量的形状为{tensor1_shape:?}，第二个张量的形状为{tensor2_shape:?}"
    )]
    OperatorError {
        operator: Operator,
        tensor1_shape: Vec<usize>,
        tensor2_shape: Vec<usize>,
    },

    #[error("张量列表为空")]
    EmptyList,
    #[error("张量形状不一致")]
    InconsitentShape,
    #[error("张量形状不兼容")]
    IncompatibleShape,
    #[error("交换张量时，输入的维度数至少需要2个")]
    PermuteNeedAtLeast2Dims,
    #[error("需要交换的维度必须是唯一且在[0, <张量维数>)范围内")]
    PermuteNeedUniqueAndInRange,
    #[error("除数为零")]
    DivByZero,
    #[error("作为除数的张量中存在为零元素")]
    DivByZeroElement,

    #[error("张量：未知错误")]
    UnKnown,
}
