/// 测试用函数描述
///
/// # 示例
///
/// ```
/// let dbsx - 5;
/// ```
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 1);
        assert_eq!(result, 4);
    }
}
