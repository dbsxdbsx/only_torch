use crate::tensor::Tensor;
use ndarray::Array;
use ndarray::IxDyn;

#[test]
fn test_new_scalar() {
    let tensor = Tensor::new(&[1.], &[]);
    assert_eq!(tensor.shape(), vec![]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[]), vec![1.]).unwrap()
    );

    let tensor = Tensor::new(&[1.], &[1]);
    assert_eq!(tensor.shape(), vec![1]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[1]), vec![1.]).unwrap()
    );

    let tensor = Tensor::new(&[1.], &[1, 1, 1]);
    assert_eq!(tensor.shape(), vec![1, 1, 1]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[1, 1, 1]), vec![1.]).unwrap()
    );

    let tensor = Tensor::new(&[1.], &[1, 1, 1, 1]);
    assert_eq!(tensor.shape(), vec![1, 1, 1, 1]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[1, 1, 1, 1]), vec![1.]).unwrap()
    );
}

#[test]
#[should_panic]
fn test_new_invalid_scalar() {
    let _ = Tensor::new(&[1., 2.], &[1, 1, 1]);
}

#[test]
fn test_new_vector() {
    // 向量
    let tensor = Tensor::new(&[1., 2., 3.], &[3]);
    assert_eq!(tensor.shape(), vec![3]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[3]), vec![1., 2., 3.]).unwrap()
    );
    // 行向量
    let tensor = Tensor::new(&[1., 2., 3.], &[1, 3]);
    assert_eq!(tensor.shape(), vec![1, 3]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[1, 3]), vec![1., 2., 3.]).unwrap()
    );
    // 列向量
    let tensor = Tensor::new(&[1., 2., 3.], &[3, 1]);
    assert_eq!(tensor.shape(), vec![3, 1]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[3, 1]), vec![1., 2., 3.]).unwrap()
    );
}

#[test]
fn test_new_matrix() {
    let tensor = Tensor::new(&[1., 2., 3., 4., 5., 6.], &[2, 3]);
    assert_eq!(tensor.shape(), vec![2, 3]);
    assert_eq!(
        tensor.data,
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![1., 2., 3., 4., 5., 6.]).unwrap()
    );
}

#[test]
fn test_new_higher_dimensional_tensor() {
    let tensor = Tensor::new(
        &[
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
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
        let min_val = -1.;
        let max_val = 1.;
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
        (2, vec![1., 0., 0., 1.]),
        (3, vec![1., 0., 0., 0., 1., 0., 0., 0., 1.]),
        (
            4,
            vec![
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
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
    let cases: &[&[usize]] = &[&[171, 6], &[57, 8], &[22, 2]];
    for case in cases {
        let mean = case[0] as f32;
        let std_dev = case[1] as f32;
        let shape = &[10, 50, 80, 30]; //尽量弄大点，以免误差太大
        let tensor = Tensor::new_normal(mean, std_dev, shape);

        // 检查形状
        assert_eq!(tensor.shape(), shape);
        // 检查生成的张量均值和标准差是否与预期值相近
        let eps = std_dev / 10.;
        let actual_mean = tensor.mean();
        let mean_diff = (actual_mean - mean).abs();
        assert!(
            mean_diff < eps,
            "均值不符合预期值，实际值为 {}，期望值为 {}, 误差上限为 {}",
            actual_mean,
            mean,
            eps
        );
        let actual_std_dev = tensor.std_dev();
        let std_dev_diff = (tensor.std_dev() - std_dev).abs();
        assert!(
            std_dev_diff < eps,
            "标准差不符合预期值，实际值为 {}，期望值为 {}，误差上限为 {}",
            actual_std_dev,
            std_dev,
            eps
        );
    }
}
