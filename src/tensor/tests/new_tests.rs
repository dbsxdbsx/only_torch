use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

#[test]
fn test_new_scalar() {
    let tensor = Tensor::new(&[1.0], &[]);
    assert_eq!(tensor.shape(), vec![]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[]), vec![1.0]).unwrap()
    );

    let tensor = Tensor::new(&[1.0], &[1]);
    assert_eq!(tensor.shape(), vec![1]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[1]), vec![1.0]).unwrap()
    );

    let tensor = Tensor::new(&[1.0], &[1, 1, 1]);
    assert_eq!(tensor.shape(), vec![1, 1, 1]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.0]).unwrap()
    );

    let tensor = Tensor::new(&[1.0], &[1, 1, 1, 1]);
    assert_eq!(tensor.shape(), vec![1, 1, 1, 1]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.0]).unwrap()
    );
}

#[test]
#[should_panic]
fn test_new_invalid_scalar() {
    let _ = Tensor::new(&[1.0, 2.0], &[1, 1, 1]);
}

#[test]
fn test_new_vector() {
    // 向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    assert_eq!(tensor.shape(), vec![3]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap()
    );
    // 行向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    assert_eq!(tensor.shape(), vec![1, 3]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap()
    );
    // 列向量
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]);
    assert_eq!(tensor.shape(), vec![3, 1]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[3, 1]), vec![1.0, 2.0, 3.0]).unwrap()
    );
}

#[test]
fn test_new_matrix() {
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(tensor.shape(), vec![2, 3]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()
    );
}

#[test]
fn test_new_higher_dimensional_tensor() {
    let tensor = Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0,
        ],
        &[3, 2, 3],
    );
    assert_eq!(tensor.shape(), vec![3, 2, 3]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[3, 2, 3]), (1..=18).map(|x| x as f32).collect()).unwrap()
    );
}

#[test]
fn test_new_random_tensor() {
    let shapes: &[&[usize]] = &[&[], &[1], &[1, 1], &[2], &[2, 3], &[2, 3, 4], &[2, 3, 4, 5]];

    for shape in shapes {
        let min_val = -1.0;
        let max_val = 1.0;
        let tensor = Tensor::new_random(min_val, max_val, shape);
        assert_eq!(tensor.shape(), *shape);

        for elem in tensor.data.iter() {
            assert!(*elem >= min_val && *elem <= max_val);
        }
    }
}

#[test]
fn test_new_eye() {
    let test_cases = vec![
        (2, vec![1.0, 0.0, 0.0, 1.0]),
        (3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        (
            4,
            vec![
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ],
        ),
    ];

    for (n, correct_result) in test_cases {
        let eye = Tensor::new_eye(n);
        assert_eq!(
            eye.data,
            Array::from_shape_vec(IxDyn(&[n, n]), correct_result).unwrap()
        );
    }
}

#[test]
fn test_new_eye_with_invalid_diagonal_size() {
    assert!(std::panic::catch_unwind(|| {
        let _ = Tensor::new_eye(0);
    })
    .is_err());

    assert!(std::panic::catch_unwind(|| {
        let _ = Tensor::new_eye(1);
    })
    .is_err());
}

#[test]
fn test_new_normal() {
    let mean = 0.0;
    let std_dev = 1.0;
    let shape = vec![100, 20, 30];
    let tensor = Tensor::new_normal(mean, std_dev, &shape);

    tensor.print();

    // 检查形状
    assert_eq!(tensor.shape(), shape);

    // 检查生成的张量均值和标准差是否与预期值相近
    let eps = std_dev / 100.0;
    let mean_diff = (tensor.mean() - mean).abs();
    assert!(
        mean_diff < eps,
        "均值不符合预期值，实际值为 {}，期望值为 {}",
        tensor.mean(),
        mean
    );
    let std_dev_diff = (tensor.std_dev() - std_dev).abs();
    assert!(
        std_dev_diff < eps,
        "标准差不符合预期值，实际值为 {}，期望值为 {}",
        tensor.std_dev(),
        std_dev
    );
}
