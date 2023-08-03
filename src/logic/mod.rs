//! # 逻辑推理模块
//!
//! 本模块提供一些和逻辑推理相关的操作接口
/// 测
///
/// # 示例
///
/// ```
/// use only_torch::logic::test_logic;
///
/// let result = test_logic(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn test_logic(left: usize, right: usize) -> usize {
    left + right
}
