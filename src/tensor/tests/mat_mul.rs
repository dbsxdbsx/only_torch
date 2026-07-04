use crate::tensor::Tensor;

// ==================== mat_mul_nt / mat_mul_tn（转置视图 GEMM）====================
//
// MatMul 反向已改用这两个 NT/TN 原语（转置只翻 stride、零物化拷贝）。
// 必须与「物化 transpose 后 mat_mul」逐 bit 一致，否则训练轨迹漂移。

/// `a.mat_mul_nt(b)` 必须与 `a.mat_mul(&b.transpose())` 逐 bit 一致
#[test]
fn test_mat_mul_nt_bitwise_matches_materialized_transpose() {
    for (m, k, n) in [(1usize, 4usize, 2usize), (3, 5, 7), (16, 64, 41)] {
        let a = Tensor::normal_seeded(0.0, 1.0, &[m, k], 31);
        let b = Tensor::normal_seeded(0.0, 1.0, &[n, k], 32);

        let fast = a.mat_mul_nt(&b);
        let reference = a.mat_mul(&b.transpose());

        assert_eq!(fast.shape(), &[m, n]);
        let (f, r) = (fast.to_vec(), reference.to_vec());
        for i in 0..f.len() {
            assert_eq!(
                f[i].to_bits(),
                r[i].to_bits(),
                "mat_mul_nt 与物化转置结果不一致：({m},{k},{n}) i={i} fast={} ref={}",
                f[i],
                r[i]
            );
        }
    }
}

/// `a.mat_mul_tn(b)` 必须与 `a.transpose().mat_mul(&b)` 逐 bit 一致
#[test]
fn test_mat_mul_tn_bitwise_matches_materialized_transpose() {
    for (k, m, n) in [(1usize, 4usize, 2usize), (5, 3, 7), (32, 64, 41)] {
        let a = Tensor::normal_seeded(0.0, 1.0, &[k, m], 41);
        let b = Tensor::normal_seeded(0.0, 1.0, &[k, n], 42);

        let fast = a.mat_mul_tn(&b);
        let reference = a.transpose().mat_mul(&b);

        assert_eq!(fast.shape(), &[m, n]);
        let (f, r) = (fast.to_vec(), reference.to_vec());
        for i in 0..f.len() {
            assert_eq!(
                f[i].to_bits(),
                r[i].to_bits(),
                "mat_mul_tn 与物化转置结果不一致：({k},{m},{n}) i={i} fast={} ref={}",
                f[i],
                r[i]
            );
        }
    }
}

/// NT/TN 原语对非连续输入（transpose 视图）也必须正确
///
/// MatMul 父节点的值可能来自 permute/transpose（非连续布局），
/// 转置视图 GEMM 不得依赖行主序假设。
#[test]
fn test_mat_mul_nt_tn_noncontiguous_inputs() {
    // a_nc: [2,4] 非连续（由 [4,2] 转置而来）
    let a_base = Tensor::normal_seeded(0.0, 1.0, &[4, 2], 51);
    let a_nc = a_base.transpose();
    assert!(!a_nc.is_contiguous());
    let a_c = Tensor::new(a_nc.to_vec(), &[2, 4]);

    let b = Tensor::normal_seeded(0.0, 1.0, &[3, 4], 52);

    let from_nc = a_nc.mat_mul_nt(&b);
    let from_c = a_c.mat_mul_nt(&b);
    assert_eq!(from_nc.shape(), &[2, 3]);
    let (x, y) = (from_nc.to_vec(), from_c.to_vec());
    for i in 0..x.len() {
        assert_eq!(
            x[i].to_bits(),
            y[i].to_bits(),
            "NT 非连续输入结果漂移：i={i}"
        );
    }

    // TN：非连续 self
    let s_base = Tensor::normal_seeded(0.0, 1.0, &[5, 4], 53);
    let s_nc = s_base.transpose(); // [4,5] 非连续
    let s_c = Tensor::new(s_nc.to_vec(), &[4, 5]);
    let o = Tensor::normal_seeded(0.0, 1.0, &[4, 6], 54);

    let from_nc = s_nc.mat_mul_tn(&o);
    let from_c = s_c.mat_mul_tn(&o);
    assert_eq!(from_nc.shape(), &[5, 6]);
    let (x, y) = (from_nc.to_vec(), from_c.to_vec());
    for i in 0..x.len() {
        assert_eq!(
            x[i].to_bits(),
            y[i].to_bits(),
            "TN 非连续输入结果漂移：i={i}"
        );
    }
}

/// NT/TN 形状不匹配时应 panic（与 mat_mul 相同契约）
#[test]
#[should_panic]
fn test_mat_mul_nt_shape_mismatch() {
    let a = Tensor::zeros(&[2, 3]);
    let b = Tensor::zeros(&[4, 5]); // b 列数 5 != a 列数 3
    let _ = a.mat_mul_nt(&b);
}

#[test]
fn test_mat_mul_vector_vector() {
    // 结果为标量的情况
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let b = Tensor::new(&[4.0, 5.0, 6.0], &[3, 1]);
    let result = a.mat_mul(&b);
    let expected = Tensor::new(&[32.0], &[1, 1]);
    assert_eq!(result.data, expected.data);
    // 结果为矩阵的情况
    let result = b.mat_mul(&a);
    let expected = Tensor::new(&[4.0, 8.0, 12.0, 5.0, 10.0, 15.0, 6.0, 12.0, 18.0], &[3, 3]);
    assert_eq!(result.data, expected.data);
    // 构造2个，使得结果正好等于第2个张量
    let a = Tensor::eyes(2);
    let b = Tensor::new(&[2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 3]);
    let result = a.mat_mul(&b);
    assert_eq!(result.data, b.data);
}

#[test]
fn test_mat_mul_vector_matrix() {
    let a = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let b = Tensor::new(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    let result = a.mat_mul(&b);
    let expected = Tensor::new(&[13.0, 16.0], &[1, 2]);
    assert_eq!(result.data, expected.data);
}

#[test]
fn test_mat_mul_matrix_vector() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[5.0, 6.0], &[2, 1]);
    let result = a.mat_mul(&b);
    let expected = Tensor::new(&[17.0, 39.0], &[2, 1]);
    assert_eq!(result.data, expected.data);
}

#[test]
fn test_mat_mul_matrix_matrix() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[5.0, 6.0, 7.0, 8.0, 9.0, 10.0], &[2, 3]);
    b.print();
    let result = a.mat_mul(&b);
    let expected = Tensor::new(&[21.0, 24.0, 27.0, 47.0, 54.0, 61.0], &[2, 3]);
    assert_eq!(result.data, expected.data);
}

#[test]
#[should_panic(expected = "输入的张量维度必须为2")]
fn test_mat_mul_panic_on_invalid_dimension() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let b = Tensor::new(&[1.0, 2.0], &[2]);
    a.mat_mul(&b);
}

#[test]
#[should_panic(expected = "前一个张量的列数必须等于后一个张量的行数")]
fn test_mat_mul_panic_on_invalid_shape() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    let b = Tensor::new(&[4.0, 5.0], &[2, 1]);
    a.mat_mul(&b);
}

// ==================== batched_mat_mul（3D 批量 GEMM，C1）====================
//
// 契约：单个 batch 切片与 2D `mat_mul` 逐 bit 一致（同一 GEMM 路径），
// 否则 attention 从逐 head 建图切换到 batched 路径时数值轨迹漂移。

/// `batched_mat_mul` 每个 batch 与 2D `mat_mul` 逐 bit 一致
#[test]
fn test_batched_mat_mul_bitwise_matches_per_batch_2d() {
    for (bsz, m, k, n) in [
        (1usize, 2usize, 3usize, 4usize),
        (3, 5, 7, 2),
        (8, 16, 8, 16),
    ] {
        let a = Tensor::normal_seeded(0.0, 1.0, &[bsz, m, k], 61);
        let b = Tensor::normal_seeded(0.0, 1.0, &[bsz, k, n], 62);

        let batched = a.batched_mat_mul(&b);
        assert_eq!(batched.shape(), &[bsz, m, n]);

        let (av, bv, ov) = (a.to_vec(), b.to_vec(), batched.to_vec());
        for i in 0..bsz {
            let a_i = Tensor::new(&av[i * m * k..(i + 1) * m * k], &[m, k]);
            let b_i = Tensor::new(&bv[i * k * n..(i + 1) * k * n], &[k, n]);
            let expected = a_i.mat_mul(&b_i).to_vec();
            let actual = &ov[i * m * n..(i + 1) * m * n];
            for j in 0..expected.len() {
                assert_eq!(
                    actual[j].to_bits(),
                    expected[j].to_bits(),
                    "batched_mat_mul 与逐 batch 2D 结果不一致：({bsz},{m},{k},{n}) batch={i} j={j}"
                );
            }
        }
    }
}

/// `batched_mat_mul_nt` 与「物化转置后 batched_mat_mul」逐 bit 一致
#[test]
fn test_batched_mat_mul_nt_bitwise_matches_materialized_transpose() {
    let (bsz, m, k, n) = (4usize, 5usize, 6usize, 3usize);
    let a = Tensor::normal_seeded(0.0, 1.0, &[bsz, m, k], 63);
    let b = Tensor::normal_seeded(0.0, 1.0, &[bsz, n, k], 64);

    let fast = a.batched_mat_mul_nt(&b);
    // 参考：手工物化 b 的 [B, k, n] 转置
    let bv = b.to_vec();
    let mut bt = vec![0.0f32; bsz * k * n];
    for i in 0..bsz {
        for r in 0..n {
            for c in 0..k {
                bt[i * k * n + c * n + r] = bv[i * n * k + r * k + c];
            }
        }
    }
    let reference = a.batched_mat_mul(&Tensor::new(&bt, &[bsz, k, n]));

    assert_eq!(fast.shape(), &[bsz, m, n]);
    let (f, r) = (fast.to_vec(), reference.to_vec());
    for i in 0..f.len() {
        assert_eq!(
            f[i].to_bits(),
            r[i].to_bits(),
            "batched_mat_mul_nt 与物化转置结果不一致：i={i}"
        );
    }
}

/// `batched_mat_mul_tn` 与「物化转置后 batched_mat_mul」逐 bit 一致
#[test]
fn test_batched_mat_mul_tn_bitwise_matches_materialized_transpose() {
    let (bsz, k, m, n) = (2usize, 7usize, 4usize, 5usize);
    let a = Tensor::normal_seeded(0.0, 1.0, &[bsz, k, m], 65);
    let b = Tensor::normal_seeded(0.0, 1.0, &[bsz, k, n], 66);

    let fast = a.batched_mat_mul_tn(&b);
    // 参考：手工物化 a 的 [B, m, k] 转置
    let av = a.to_vec();
    let mut at = vec![0.0f32; bsz * m * k];
    for i in 0..bsz {
        for r in 0..k {
            for c in 0..m {
                at[i * m * k + c * k + r] = av[i * k * m + r * m + c];
            }
        }
    }
    let reference = Tensor::new(&at, &[bsz, m, k]).batched_mat_mul(&b);

    assert_eq!(fast.shape(), &[bsz, m, n]);
    let (f, r) = (fast.to_vec(), reference.to_vec());
    for i in 0..f.len() {
        assert_eq!(
            f[i].to_bits(),
            r[i].to_bits(),
            "batched_mat_mul_tn 与物化转置结果不一致：i={i}"
        );
    }
}

/// `batched_mat_mul` 对非连续输入（permute 视图）也必须正确
#[test]
fn test_batched_mat_mul_noncontiguous_inputs() {
    // 由 [k, B, m] permute 成 [B, m, k] 的非连续视图
    let base = Tensor::normal_seeded(0.0, 1.0, &[3, 2, 4], 67); // [k=3, B=2, m=4]
    let a_nc = base.permute(&[1, 2, 0]); // [B=2, m=4, k=3]
    assert!(!a_nc.is_contiguous());
    let a_c = Tensor::new(a_nc.to_vec(), &[2, 4, 3]);

    let b = Tensor::normal_seeded(0.0, 1.0, &[2, 3, 5], 68);

    let from_nc = a_nc.batched_mat_mul(&b);
    let from_c = a_c.batched_mat_mul(&b);
    assert_eq!(from_nc.shape(), &[2, 4, 5]);
    let (x, y) = (from_nc.to_vec(), from_c.to_vec());
    for i in 0..x.len() {
        assert_eq!(
            x[i].to_bits(),
            y[i].to_bits(),
            "batched_mat_mul 非连续输入结果漂移：i={i}"
        );
    }
}

/// batch 维不相等时应 panic（不做跨 batch 隐式广播）
#[test]
#[should_panic(expected = "batch 维必须严格相等")]
fn test_batched_mat_mul_panic_on_batch_mismatch() {
    let a = Tensor::zeros(&[2, 3, 4]);
    let b = Tensor::zeros(&[3, 4, 5]);
    let _ = a.batched_mat_mul(&b);
}

/// 内维不匹配时应 panic
#[test]
#[should_panic(expected = "内维不匹配")]
fn test_batched_mat_mul_panic_on_inner_mismatch() {
    let a = Tensor::zeros(&[2, 3, 4]);
    let b = Tensor::zeros(&[2, 5, 6]);
    let _ = a.batched_mat_mul(&b);
}

/// 2D 输入应 panic（batched 版只接受 3D）
#[test]
#[should_panic(expected = "必须为 3D")]
fn test_batched_mat_mul_panic_on_2d_input() {
    let a = Tensor::zeros(&[3, 4]);
    let b = Tensor::zeros(&[4, 5]);
    let _ = a.batched_mat_mul(&b);
}
