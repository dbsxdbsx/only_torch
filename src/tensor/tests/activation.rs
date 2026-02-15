/*
 * @Author       : 老董
 * @Date         : 2026-02-14
 * @Description  : Tensor 激活函数测试：relu、leaky_relu 等
 */

use crate::tensor::Tensor;

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓relu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_relu_basic() {
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = x.relu();
    let expected = [0.0, 0.0, 0.0, 1.0, 2.0];
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (y[[i]] - e).abs() < 1e-6,
            "relu index {i}: got {}, expected {e}",
            y[[i]]
        );
    }
}

#[test]
fn test_relu_2d() {
    let x = Tensor::new(&[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], &[2, 3]);
    let y = x.relu();
    assert_eq!(y.shape(), &[2, 3]);
    assert!((y[[0, 0]] - 0.0).abs() < 1e-6); // -1.0 → 0
    assert!((y[[0, 1]] - 2.0).abs() < 1e-6); // 2.0 → 2.0
    assert!((y[[1, 2]] - 6.0).abs() < 1e-6); // 6.0 → 6.0
}

#[test]
fn test_relu_matches_leaky_relu_zero() {
    // relu(x) 应与 leaky_relu(x, 0.0) 结果一致
    let x = Tensor::new(&[-3.0, -1.0, 0.0, 1.0, 3.0], &[5]);
    let y_relu = x.relu();
    let y_leaky = x.leaky_relu(0.0);
    for i in 0..5 {
        assert!(
            (y_relu[[i]] - y_leaky[[i]]).abs() < 1e-6,
            "relu 与 leaky_relu(0.0) 不一致，index {i}: relu={}, leaky={}",
            y_relu[[i]],
            y_leaky[[i]]
        );
    }
}

#[test]
fn test_relu_no_nan() {
    let x = Tensor::new(&[f32::MAX, f32::MIN, 0.0, -0.0], &[4]);
    let y = x.relu();
    for i in 0..4 {
        assert!(!y[[i]].is_nan(), "relu 产生了 NaN，index {i}");
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑relu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓leaky_relu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_leaky_relu_basic() {
    // 基本值正确性
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = x.leaky_relu(0.1);
    let expected = [-0.2, -0.1, 0.0, 1.0, 2.0];
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (y[[i]] - e).abs() < 1e-6,
            "leaky_relu index {i}: got {}, expected {e}",
            y[[i]]
        );
    }
}

#[test]
fn test_leaky_relu_as_relu() {
    // alpha=0 退化为标准 ReLU
    let x = Tensor::new(&[-3.0, -1.0, 0.0, 1.0, 3.0], &[5]);
    let y = x.leaky_relu(0.0);
    let expected = [0.0, 0.0, 0.0, 1.0, 3.0];
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (y[[i]] - e).abs() < 1e-6,
            "relu index {i}: got {}, expected {e}",
            y[[i]]
        );
    }
}

#[test]
fn test_leaky_relu_2d() {
    // 2D 张量，验证形状保持
    let x = Tensor::new(&[-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], &[2, 3]);
    let y = x.leaky_relu(0.01);
    assert_eq!(y.shape(), &[2, 3]);
    assert!((y[[0, 0]] - (-0.01)).abs() < 1e-6);
    assert!((y[[0, 1]] - 2.0).abs() < 1e-6);
    assert!((y[[1, 2]] - 6.0).abs() < 1e-6);
}

#[test]
fn test_leaky_relu_extreme_values() {
    // 边界值：极大、极小、接近零
    let x = Tensor::new(&[1e10, -1e10, 1e-10, -1e-10], &[4]);
    let y = x.leaky_relu(0.1);
    assert!((y[[0]] - 1e10).abs() < 1.0);
    assert!((y[[1]] - (-1e9)).abs() < 1.0); // -1e10 * 0.1 = -1e9
    assert!((y[[2]] - 1e-10).abs() < 1e-15); // 正的微小值不变
    assert!((y[[3]] - (-1e-11)).abs() < 1e-15); // 负的微小值 * 0.1
}

#[test]
fn test_leaky_relu_no_nan() {
    // 数值稳定性：不产生 NaN/Inf
    let x = Tensor::new(&[f32::MAX, f32::MIN, 0.0, -0.0], &[4]);
    let y = x.leaky_relu(0.01);
    for i in 0..4 {
        assert!(!y[[i]].is_nan(), "leaky_relu 产生了 NaN，index {i}");
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑leaky_relu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓softplus↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_softplus_basic() {
    // softplus(0) = ln(2) ≈ 0.6931
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = x.softplus();
    // Python: torch.nn.functional.softplus(torch.tensor([-2,-1,0,1,2]))
    let expected = [0.1269, 0.3133, 0.6931, 1.3133, 2.1269];
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (y[[i]] - e).abs() < 1e-3,
            "softplus index {i}: got {}, expected {e}",
            y[[i]]
        );
    }
}

#[test]
fn test_softplus_numerical_stability() {
    // 大正数不溢出，大负数不下溢到负数
    let x = Tensor::new(&[50.0, 100.0, -50.0, -100.0], &[4]);
    let y = x.softplus();
    // 大正数：softplus(x) ≈ x
    assert!((y[[0]] - 50.0).abs() < 1e-6);
    assert!((y[[1]] - 100.0).abs() < 1e-6);
    // 大负数：softplus(x) ≈ 0
    assert!(y[[2]] >= 0.0 && y[[2]] < 1e-10);
    assert!(y[[3]] >= 0.0 && y[[3]] < 1e-10);
    // 不产生 NaN/Inf
    for i in 0..4 {
        assert!(!y[[i]].is_nan(), "softplus 产生了 NaN，index {i}");
        assert!(!y[[i]].is_infinite(), "softplus 产生了 Inf，index {i}");
    }
}

#[test]
fn test_softplus_always_positive() {
    // softplus 输出恒正
    let x = Tensor::new(&[-10.0, -1.0, 0.0, 1.0, 10.0], &[5]);
    let y = x.softplus();
    for i in 0..5 {
        assert!(y[[i]] > 0.0, "softplus 应恒正，index {i}: got {}", y[[i]]);
    }
}

#[test]
fn test_softplus_shape_preserved() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = x.softplus();
    assert_eq!(y.shape(), &[2, 3]);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑softplus↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓step_fn↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_step_fn_basic() {
    let x = Tensor::new(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5]);
    let y = x.step_fn();
    let expected = [0.0, 0.0, 1.0, 1.0, 1.0];
    for (i, &e) in expected.iter().enumerate() {
        assert!(
            (y[[i]] - e).abs() < 1e-6,
            "step_fn index {i}: got {}, expected {e}",
            y[[i]]
        );
    }
}

#[test]
fn test_step_fn_extreme_values() {
    let x = Tensor::new(&[f32::MAX, f32::MIN, 1e-30, -1e-30], &[4]);
    let y = x.step_fn();
    assert!((y[[0]] - 1.0).abs() < 1e-6); // MAX -> 1
    assert!((y[[1]] - 0.0).abs() < 1e-6); // MIN -> 0
    assert!((y[[2]] - 1.0).abs() < 1e-6); // 微小正数 -> 1
    assert!((y[[3]] - 0.0).abs() < 1e-6); // 微小负数 -> 0
}

#[test]
fn test_step_fn_shape_preserved() {
    let x = Tensor::new(&[-1.0, 1.0, -1.0, 1.0, -1.0, 1.0], &[2, 3]);
    let y = x.step_fn();
    assert_eq!(y.shape(), &[2, 3]);
}

#[test]
fn test_step_fn_no_nan() {
    let x = Tensor::new(&[0.0, -0.0, f32::MAX, f32::MIN], &[4]);
    let y = x.step_fn();
    for i in 0..4 {
        assert!(!y[[i]].is_nan(), "step_fn 产生了 NaN，index {i}");
    }
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑step_fn↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓gelu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_gelu_basic() {
    // PyTorch tanh approximate: gelu([0.5, -1.0, 0.0, 2.0])
    let x = Tensor::new(&[0.5, -1.0, 0.0, 2.0], &[4]);
    let y = x.gelu();
    assert!((y[[0]] - 0.3457).abs() < 1e-3);
    assert!((y[[1]] - (-0.1588)).abs() < 1e-3);
    assert!((y[[2]] - 0.0).abs() < 1e-6);
    assert!((y[[3]] - 1.9546).abs() < 1e-3);
}

#[test]
fn test_gelu_symmetry() {
    // gelu(0) = 0 精确
    let x = Tensor::new(&[0.0], &[1]);
    let y = x.gelu();
    assert!((y[[0]]).abs() < 1e-8);
}

#[test]
fn test_gelu_large_values() {
    // 大正数 gelu(x) ≈ x，大负数 gelu(x) ≈ 0
    let x = Tensor::new(&[20.0, -20.0], &[2]);
    let y = x.gelu();
    assert!((y[[0]] - 20.0).abs() < 1e-4);
    assert!(y[[1]].abs() < 1e-4);
}

#[test]
fn test_gelu_shape_preserved() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let y = x.gelu();
    assert_eq!(y.shape(), &[2, 3]);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑gelu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓swish↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_swish_basic() {
    // swish(x) = x * sigmoid(x)
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = x.swish();
    // Python: x * torch.sigmoid(x)
    assert!((y[[2]] - 0.0).abs() < 1e-6); // swish(0) = 0
    assert!((y[[3]] - 0.7311).abs() < 1e-3); // swish(1) = 1*sigmoid(1)
    assert!(y[[0]] < 0.0); // swish(-2) < 0
}

#[test]
fn test_swish_shape_preserved() {
    let x = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    assert_eq!(x.swish().shape(), &[2, 3]);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑swish↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓elu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_elu_basic() {
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let y = x.elu(1.0);
    assert!((y[[0]] - (-0.8647)).abs() < 1e-3); // alpha*(exp(-2)-1)
    assert!((y[[2]] - 0.0).abs() < 1e-6); // elu(0) = 0
    assert!((y[[3]] - 1.0).abs() < 1e-6); // elu(1) = 1
    assert!((y[[4]] - 2.0).abs() < 1e-6); // elu(2) = 2
}

#[test]
fn test_elu_custom_alpha() {
    let x = Tensor::new(&[-1.0, 0.0, 1.0], &[3]);
    let y = x.elu(0.5);
    assert!((y[[0]] - 0.5 * ((-1.0_f32).exp() - 1.0)).abs() < 1e-6);
    assert!((y[[1]] - 0.0).abs() < 1e-6);
    assert!((y[[2]] - 1.0).abs() < 1e-6);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑elu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓selu↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_selu_basic() {
    let x = Tensor::new(&[-1.0, 0.0, 1.0], &[3]);
    let y = x.selu();
    // Python: torch.nn.functional.selu(torch.tensor([-1,0,1]))
    assert!((y[[1]] - 0.0).abs() < 1e-6); // selu(0) = 0
    assert!((y[[2]] - 1.0507).abs() < 1e-3); // selu(1) = LAMBDA * 1
    assert!(y[[0]] < 0.0); // selu(-1) < 0
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑selu↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓mish↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_mish_basic() {
    let x = Tensor::new(&[-2.0, 0.0, 1.0, 5.0], &[4]);
    let y = x.mish();
    assert!((y[[1]] - 0.0).abs() < 1e-6); // mish(0) = 0
    assert!(y[[2] ] > 0.0); // mish(1) > 0
    assert!((y[[3]] - 5.0).abs() < 0.01); // mish(5) ≈ 5
    assert!(y[[0]] < 0.0); // mish(-2) < 0（slight negative）
}

#[test]
fn test_mish_numerical_stability() {
    let x = Tensor::new(&[50.0, -50.0], &[2]);
    let y = x.mish();
    assert!((y[[0]] - 50.0).abs() < 1e-4); // mish(50) ≈ 50
    assert!(y[[1]].abs() < 1e-4); // mish(-50) ≈ 0
    assert!(!y[[0]].is_nan());
    assert!(!y[[1]].is_nan());
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑mish↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓hard_swish↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_hard_swish_basic() {
    let x = Tensor::new(&[-4.0, -3.0, 0.0, 3.0, 4.0], &[5]);
    let y = x.hard_swish();
    assert!((y[[0]] - 0.0).abs() < 1e-6); // x <= -3: 0
    assert!((y[[1]] - 0.0).abs() < 1e-6); // x = -3: 0
    assert!((y[[2]] - 0.0).abs() < 1e-6); // x = 0: 0*(0+3)/6 = 0
    assert!((y[[3]] - 3.0).abs() < 1e-6); // x >= 3: x
    assert!((y[[4]] - 4.0).abs() < 1e-6); // x >= 3: x
}

#[test]
fn test_hard_swish_middle() {
    // 中间段：x*(x+3)/6
    let x = Tensor::new(&[1.0], &[1]);
    let y = x.hard_swish();
    assert!((y[[0]] - 1.0 * (1.0 + 3.0) / 6.0).abs() < 1e-6);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑hard_swish↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓hard_sigmoid↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_hard_sigmoid_basic() {
    let x = Tensor::new(&[-4.0, -3.0, 0.0, 3.0, 4.0], &[5]);
    let y = x.hard_sigmoid();
    assert!((y[[0]] - 0.0).abs() < 1e-6); // x <= -3: 0
    assert!((y[[1]] - 0.0).abs() < 1e-6); // x = -3: 0
    assert!((y[[2]] - 0.5).abs() < 1e-6); // x = 0: (0+3)/6 = 0.5
    assert!((y[[3]] - 1.0).abs() < 1e-6); // x >= 3: 1
    assert!((y[[4]] - 1.0).abs() < 1e-6); // x >= 3: 1
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑hard_sigmoid↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓square↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_square() {
    let x = Tensor::new(&[-3.0, -2.0, 0.0, 1.0, 2.0, 3.0], &[2, 3]);
    let y = x.square();
    assert_eq!(y.shape(), &[2, 3]);
    assert!((y[[0, 0]] - 9.0).abs() < 1e-6);
    assert!((y[[0, 1]] - 4.0).abs() < 1e-6);
    assert!((y[[0, 2]] - 0.0).abs() < 1e-6);
    assert!((y[[1, 0]] - 1.0).abs() < 1e-6);
    assert!((y[[1, 1]] - 4.0).abs() < 1e-6);
    assert!((y[[1, 2]] - 9.0).abs() < 1e-6);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑square↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓reciprocal↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_reciprocal() {
    let x = Tensor::new(&[1.0, 2.0, 4.0, 5.0, 0.5, 10.0], &[2, 3]);
    let y = x.reciprocal();
    assert_eq!(y.shape(), &[2, 3]);
    assert!((y[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((y[[0, 1]] - 0.5).abs() < 1e-6);
    assert!((y[[0, 2]] - 0.25).abs() < 1e-6);
    assert!((y[[1, 0]] - 0.2).abs() < 1e-6);
    assert!((y[[1, 1]] - 2.0).abs() < 1e-6);
    assert!((y[[1, 2]] - 0.1).abs() < 1e-6);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑reciprocal↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓log10↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_log10() {
    let x = Tensor::new(&[1.0, 10.0, 100.0, 1000.0], &[2, 2]);
    let y = x.log10();
    assert!((y[[0, 0]] - 0.0).abs() < 1e-6);
    assert!((y[[0, 1]] - 1.0).abs() < 1e-6);
    assert!((y[[1, 0]] - 2.0).abs() < 1e-5);
    assert!((y[[1, 1]] - 3.0).abs() < 1e-5);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑log10↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓log2↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_log2() {
    let x = Tensor::new(&[1.0, 2.0, 4.0, 8.0], &[2, 2]);
    let y = x.log2();
    assert!((y[[0, 0]] - 0.0).abs() < 1e-6);
    assert!((y[[0, 1]] - 1.0).abs() < 1e-6);
    assert!((y[[1, 0]] - 2.0).abs() < 1e-5);
    assert!((y[[1, 1]] - 3.0).abs() < 1e-5);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑log2↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓relu6↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_relu6() {
    let x = Tensor::new(&[-2.0, -1.0, 0.0, 3.0, 6.0, 7.0], &[2, 3]);
    let y = x.relu6();
    assert!((y[[0, 0]] - 0.0).abs() < 1e-6);
    assert!((y[[0, 1]] - 0.0).abs() < 1e-6);
    assert!((y[[0, 2]] - 0.0).abs() < 1e-6);
    assert!((y[[1, 0]] - 3.0).abs() < 1e-6);
    assert!((y[[1, 1]] - 6.0).abs() < 1e-6);
    assert!((y[[1, 2]] - 6.0).abs() < 1e-6);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑relu6↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓hard_tanh↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_hard_tanh() {
    let x = Tensor::new(&[-2.0, -0.5, 0.5, 2.0], &[2, 2]);
    let y = x.hard_tanh(-1.0, 1.0);
    assert!((y[[0, 0]] - (-1.0)).abs() < 1e-6);
    assert!((y[[0, 1]] - (-0.5)).abs() < 1e-6);
    assert!((y[[1, 0]] - 0.5).abs() < 1e-6);
    assert!((y[[1, 1]] - 1.0).abs() < 1e-6);
}

#[test]
fn test_tensor_hard_tanh_custom_range() {
    let x = Tensor::new(&[-5.0, 0.0, 3.0, 10.0], &[2, 2]);
    let y = x.hard_tanh(-3.0, 5.0);
    assert!((y[[0, 0]] - (-3.0)).abs() < 1e-6);
    assert!((y[[0, 1]] - 0.0).abs() < 1e-6);
    assert!((y[[1, 0]] - 3.0).abs() < 1e-6);
    assert!((y[[1, 1]] - 5.0).abs() < 1e-6);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑hard_tanh↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

/*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓one_hot↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
#[test]
fn test_tensor_one_hot() {
    let indices = Tensor::new(&[0.0, 2.0, 1.0], &[3]);
    let encoded = indices.one_hot(3);
    assert_eq!(encoded.shape(), &[3, 3]);
    // [0] → [1, 0, 0]
    assert!((encoded[[0, 0]] - 1.0).abs() < 1e-6);
    assert!((encoded[[0, 1]] - 0.0).abs() < 1e-6);
    assert!((encoded[[0, 2]] - 0.0).abs() < 1e-6);
    // [2] → [0, 0, 1]
    assert!((encoded[[1, 0]] - 0.0).abs() < 1e-6);
    assert!((encoded[[1, 2]] - 1.0).abs() < 1e-6);
    // [1] → [0, 1, 0]
    assert!((encoded[[2, 1]] - 1.0).abs() < 1e-6);
}

#[test]
fn test_tensor_one_hot_2d() {
    let indices = Tensor::new(&[0.0, 1.0, 2.0, 0.0], &[2, 2]);
    let encoded = indices.one_hot(3);
    assert_eq!(encoded.shape(), &[2, 2, 3]);
    // [0,0]=0 → [1,0,0]
    assert!((encoded[[0, 0, 0]] - 1.0).abs() < 1e-6);
    // [0,1]=1 → [0,1,0]
    assert!((encoded[[0, 1, 1]] - 1.0).abs() < 1e-6);
    // [1,0]=2 → [0,0,1]
    assert!((encoded[[1, 0, 2]] - 1.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "one_hot")]
fn test_tensor_one_hot_index_out_of_range() {
    let indices = Tensor::new(&[0.0, 5.0], &[2]);
    let _encoded = indices.one_hot(3);
}
/*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑one_hot↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
