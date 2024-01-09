use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

use super::TensorCheck;
#[cfg(test)]
mod tests {
    use super::Tensor;

    #[test]
    fn test_mat_mul_vector() {
        // 1阶向量的矩阵乘法
        let a = Tensor::new(&[1.0, 2.0], &[2]);
        let b = Tensor::new(&[3.0, 4.0], &[2]);
        let result = a.mat_mul(&b);
        let expected = Tensor::new(&[11.0], &[]); // 形状对吗？
        assert_eq!(result.data, expected.data);

        // 2阶向量的矩阵乘法
        let a = Tensor::new(&[1.0, 2.0], &[1, 2]);
        let b = Tensor::new(&[3.0, 4.0], &[2, 1]);
        let result = a.mat_mul(&b);
        let expected = Tensor::new(&[11.0], &[1, 1]); // 形状对吗？
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_mat_mul_matrix() {
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let result = a.mat_mul(&b);
        let expected = Tensor::new(&[19.0, 22.0, 43.0, 50.0], &[2, 2]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_mat_mul_vector_matrix() {
        let a = Tensor::new(&[1.0, 2.0], &[2]);
        let b = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
        let result = a.mat_mul(&b);
        let expected = Tensor::new(&[13.0, 16.0], &[2]);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    #[should_panic(expected = "输入的张量维度必须为1或2")]
    fn test_mat_mul_panic_on_invalid_dims() {
        let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let b = Tensor::new(&[1.0, 2.0], &[2]);
        a.mat_mul(&b);
    }

    #[test]
    #[should_panic(expected = "输入的张量维度必须为1或2")]
    fn test_mat_mul_panic_on_invalid_shape() {
        let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let b = Tensor::new(&[4.0, 5.0], &[2]);
        a.mat_mul(&b);
    }
}
