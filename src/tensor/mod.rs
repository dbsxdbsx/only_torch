mod tensor;
pub use tensor::Tensor;

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;

    #[test]
    fn test_empty() {
        // 测试创建一个2x3的空Tensor，数据全部为 0.0
        let tensor = Tensor::zero(2, 3);
        assert_eq!(
            tensor.data(),
            &na::DMatrix::from_row_slice(2, 3, &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        );

        // 测试创建一个1x3的空Tensor，数据全部为 0.0
        let tensor = Tensor::zero(1, 3);
        assert_eq!(
            tensor.data(),
            &na::DMatrix::from_row_slice(1, 3, &[0.0, 0.0, 0.0])
        );

        // 测试创建一个2x1的空Tensor，数据全部为 0.0
        let tensor = Tensor::zero(2, 1);
        assert_eq!(
            tensor.data(),
            &na::DMatrix::from_row_slice(2, 1, &[0.0, 0.0])
        );
    }

    #[test]
    fn test_scalar() {
        // 先检查下scalar()方法
        let value: f32 = 42.0;
        let scalar_tensor = Tensor::scalar(value);
        let data = scalar_tensor.data();
        assert_eq!(scalar_tensor.shape(), &[1, 1]);
        assert_eq!(data.index((0, 0)), &value);

        // 再检查下to_scalar()方法
        let ret_value = scalar_tensor.to_scalar().unwrap();
        assert!(ret_value - value < 1e-6);
    }

    #[test]
    fn test_random() {
        let tensor = Tensor::random(2, 3, 0.0, 1.0);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert!(tensor.data().iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn test_eye() {
        let tensor = Tensor::eye(1);
        assert_eq!(tensor.data(), &na::DMatrix::from_row_slice(1, 1, &[1.0]));

        let tensor = Tensor::eye(3);
        assert_eq!(
            tensor.data(),
            &na::DMatrix::from_row_slice(3, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        );
    }

    #[test]
    fn test_tensor_mul_scalar() {
        let tensor1 = Tensor::random(2, 2, 0.0, 1.0);
        let scalar = 2.0;
        let tensor2 = Tensor::scalar(scalar);

        let result1 = tensor1.clone() * scalar;
        let result2 = scalar * tensor1.clone();
        let result3 = tensor1.clone() * tensor2.clone();
        // let result4 = tensor2 * tensor1.clone();
        println!("======================");
        println!("result1: {}", result1);
        println!("result2: {}", result2);
        println!("result3: {}", result3);
        // println!("result4: {}", result4);
        println!("======================");
        let epsilon = 1e-6;
        for i in 0..result1.shape()[0] {
            for j in 0..result1.shape()[1] {
                assert!(
                    (result1.data().index((i, j)) - result2.data().index((i, j))).abs() < epsilon
                );
                assert!(
                    (result1.data().index((i, j)) - result3.data().index((i, j))).abs() < epsilon
                );
                // assert!(
                //     (result1.data().index((i, j)) - result4.data().index((i, j))).abs() < epsilon
                // );
            }
        }
    }

    // 这个测试肯定是通过的，但需要通过手动执行来查看确认打印结果
    #[test]
    fn test_print() {
        let tensor = Tensor::eye(1);
        println!("{}", tensor);
        let tensor = Tensor::eye(2);
        println!("{}", tensor);
        let tensor = Tensor::eye(3);
        println!("{}", tensor);
        let tensor = Tensor::eye(7);
        println!("{}", tensor);
        let tensor = Tensor::random(2, 2, 0.0, 1.0);
        println!("{}", tensor);
        let tensor = Tensor::random(7, 4, 0.0, 1.0);
        println!("{}", tensor);
        let tensor = Tensor::random(4, 7, 0.0, 1.0);
        println!("{}", tensor);
        let tensor = Tensor::random(1, 7, 0.0, 1.0);
        println!("{}", tensor);
        let tensor = Tensor::random(7, 1, 0.0, 1.0);
        println!("{}", tensor);
        let tensor = Tensor::random(6, 6, 0.0, 1.0);
        println!("{}", tensor);
    }
}
