//! # 常用接口模块
//!
//! 本模块提供一些常用的操作接口
/// 测
///
/// # 示例
///
/// ```
/// use only_torch::utils::test_utils;
///
/// let result = test_utils(2, 3);
/// assert_eq!(result, 5);
/// ```
pub fn test_utils(left: usize, right: usize) -> usize {
    left + right
}
