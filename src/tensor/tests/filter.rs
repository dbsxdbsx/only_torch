#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_gt() {
        // 标量测试
        let t = Tensor::new(&[1.0], &[]);
        let result = t.where_greater_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0], &[]);
        assert_eq!(result, expected);

        // 向量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1], &[3]);
        let result = t.where_greater_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, -2.0, 2.0], &[3]);
        assert_eq!(result, expected);

        // 矩阵测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[2, 2]);
        let result = t.where_greater_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, -2.0, 2.0, 2.0], &[2, 2]);
        assert_eq!(result, expected);

        // 高维张量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2, 0.8, 1.0, 1.3, 0.7], &[2, 2, 2]);
        let result = t.where_greater_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, -2.0, 2.0, 2.0, -2.0, -2.0, 2.0, -2.0], &[2, 2, 2]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_lt() {
        // 标量测试
        let t = Tensor::new(&[1.0], &[]);
        let result = t.where_lower_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0], &[]);
        assert_eq!(result, expected);

        // 向量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1], &[3]);
        let result = t.where_lower_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, -2.0, -2.0], &[3]);
        assert_eq!(result, expected);

        // 矩阵测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[2, 2]);
        let result = t.where_lower_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, -2.0, -2.0, -2.0], &[2, 2]);
        assert_eq!(result, expected);

        // 高维张量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2, 0.8, 1.0, 1.3, 0.7], &[2, 2, 2]);
        let result = t.where_lower_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, -2.0, -2.0, -2.0, 2.0, -2.0, -2.0, 2.0], &[2, 2, 2]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ge() {
        // 标量测试
        let t = Tensor::new(&[1.0], &[]);
        let result = t.where_greater_equal_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0], &[]);
        assert_eq!(result, expected);

        // 向量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1], &[3]);
        let result = t.where_greater_equal_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, 2.0, 2.0], &[3]);
        assert_eq!(result, expected);

        // 矩阵测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[2, 2]);
        let result = t.where_greater_equal_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, 2.0, 2.0, 2.0], &[2, 2]);
        assert_eq!(result, expected);

        // 高维张量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2, 0.8, 1.0, 1.3, 0.7], &[2, 2, 2]);
        let result = t.where_greater_equal_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0, -2.0], &[2, 2, 2]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_le() {
        // 标量测试
        let t = Tensor::new(&[1.0], &[]);
        let result = t.where_lower_equal_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0], &[]);
        assert_eq!(result, expected);

        // 向量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1], &[3]);
        let result = t.where_lower_equal_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, 2.0, -2.0], &[3]);
        assert_eq!(result, expected);

        // 矩阵测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[2, 2]);
        let result = t.where_lower_equal_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, 2.0, -2.0, -2.0], &[2, 2]);
        assert_eq!(result, expected);

        // 高维张量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2, 0.8, 1.0, 1.3, 0.7], &[2, 2, 2]);
        let result = t.where_lower_equal_than(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, 2.0, -2.0, -2.0, 2.0, 2.0, -2.0, 2.0], &[2, 2, 2]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eq() {
        // 标量测试
        let t = Tensor::new(&[1.0], &[]);
        let result = t.where_equal(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0], &[]);
        assert_eq!(result, expected);

        // 向量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1], &[3]);
        let result = t.where_equal(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, 2.0, -2.0], &[3]);
        assert_eq!(result, expected);

        // 矩阵测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[2, 2]);
        let result = t.where_equal(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, 2.0, -2.0, -2.0], &[2, 2]);
        assert_eq!(result, expected);

        // 高维张量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2, 0.8, 1.0, 1.3, 0.7], &[2, 2, 2]);
        let result = t.where_equal(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0, 2.0, -2.0, -2.0, -2.0, 2.0, -2.0, -2.0], &[2, 2, 2]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ne() {
        // 标量测试
        let t = Tensor::new(&[1.0], &[]);
        let result = t.where_not_equal(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[-2.0], &[]);
        assert_eq!(result, expected);

        // 向量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1], &[3]);
        let result = t.where_not_equal(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, -2.0, 2.0], &[3]);
        assert_eq!(result, expected);

        // 矩阵测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2], &[2, 2]);
        let result = t.where_not_equal(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, -2.0, 2.0, 2.0], &[2, 2]);
        assert_eq!(result, expected);

        // 高维张量测试
        let t = Tensor::new(&[0.9, 1.0, 1.1, 1.2, 0.8, 1.0, 1.3, 0.7], &[2, 2, 2]);
        let result = t.where_not_equal(1.0, 2.0, -2.0);
        let expected = Tensor::new(&[2.0, -2.0, 2.0, 2.0, 2.0, -2.0, 2.0, 2.0], &[2, 2, 2]);
        assert_eq!(result, expected);
    }

    // 测试边界值
    #[test]
    fn test_boundary_values() {
        let t = Tensor::new(&[f32::MIN, 0.0, f32::MAX], &[3]);
        let result = t.where_greater_than(0.0, 1.0, 0.0);
        assert_eq!(result.data.as_slice().unwrap(), &[0.0, 0.0, 1.0]);
    }

    // 测试特殊值
    #[test]
    fn test_special_values() {
        let t = Tensor::new(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], &[3]);
        let result = t.where_greater_than(0.0, 2.0, -2.0);
        let result_slice = result.data.as_slice().unwrap();

        // 对于 NaN，我们需要单独检查
        assert!(result_slice[0].is_nan());
        // 对于其他值，我们可以正常比较
        assert_eq!(result_slice[1], 2.0); // INFINITY > 0
        assert_eq!(result_slice[2], -2.0); // NEG_INFINITY < 0
    }
}
