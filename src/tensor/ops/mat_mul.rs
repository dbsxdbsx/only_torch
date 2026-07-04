use crate::tensor::{Tensor, next_source_id};

/// 把 GEMM 输出统一为标准（C 连续）布局的动态维数组
///
/// ndarray 0.16 起 BLAS 派发覆盖全部兼容布局（#1419），转置视图输入的
/// `dot`（如 TN 形式 `A^T @ B`）可能直接以 **F 序**输出；而本框架约定
/// `Tensor` 运算产物均为标准布局（`flatten_view` / `data_as_slice` 等依赖）。
/// 已是标准布局时零开销直通，仅 F 序输出付一次物化。
fn into_standard_dyn(result: ndarray::Array2<f32>) -> ndarray::ArrayD<f32> {
    if result.is_standard_layout() {
        result.into_dyn()
    } else {
        result.as_standard_layout().into_owned().into_dyn()
    }
}

impl Tensor {
    /// 实现矩阵乘法(`mat_mul`这个名称参考了python的`numpy`库)。
    /// 只接受2维张量(即矩阵)，否则会触发panic。
    /// 需要保证前一个张量的列数（col）等于后一个张量的行数（row），否则也会触发panic。
    pub fn mat_mul(&self, other: &Self) -> Self {
        // 检查输入的张量维度
        let self_dims = self.dimension();
        let other_dims = other.dimension();
        assert!(self_dims == 2, "输入的张量维度必须为2");
        assert!(other_dims == 2, "输入的张量维度必须为2");
        // 检查前一个张量的列数是否等于后一个张量的行数
        assert!(
            self.shape()[1] == other.shape()[0],
            "前一个张量的列数必须等于后一个张量的行数"
        );
        // 将动态维度数组转换为常量维度数组
        let self_data = self
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let other_data = other
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        // 执行矩阵乘法
        let result_data = self_data.dot(&other_data);
        // 创建并返回新的张量
        Self {
            data: into_standard_dyn(result_data).into_shared(),
            source_id: next_source_id(),
        }
    }

    /// 矩阵乘法 `self @ other^T`（NT 形式，other 以**转置视图**参与，零物化拷贝）
    ///
    /// 要求 `self: [m, k]`、`other: [n, k]`，输出 `[m, n]`。
    /// 等价于 `self.mat_mul(&other.transpose())`，但转置只翻转 stride 元数据、
    /// 不复制数据（ndarray `dot` 原生支持转置视图，BLAS 路径走转置标志）。
    /// 典型用途：MatMul 反向 `dL/dA = upstream @ B^T`。
    pub fn mat_mul_nt(&self, other: &Self) -> Self {
        assert!(self.dimension() == 2, "输入的张量维度必须为2");
        assert!(other.dimension() == 2, "输入的张量维度必须为2");
        assert!(
            self.shape()[1] == other.shape()[1],
            "mat_mul_nt: self 的列数 {} 必须等于 other 的列数 {}（other 以转置参与）",
            self.shape()[1],
            other.shape()[1]
        );
        let self_data = self
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let other_data = other
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let result_data = self_data.dot(&other_data.t());
        Self {
            data: into_standard_dyn(result_data).into_shared(),
            source_id: next_source_id(),
        }
    }

    /// 矩阵乘法 `self^T @ other`（TN 形式，self 以**转置视图**参与，零物化拷贝）
    ///
    /// 要求 `self: [k, m]`、`other: [k, n]`，输出 `[m, n]`。
    /// 等价于 `self.transpose().mat_mul(other)`，但转置零拷贝。
    /// 典型用途：MatMul 反向 `dL/dB = A^T @ upstream`。
    pub fn mat_mul_tn(&self, other: &Self) -> Self {
        assert!(self.dimension() == 2, "输入的张量维度必须为2");
        assert!(other.dimension() == 2, "输入的张量维度必须为2");
        assert!(
            self.shape()[0] == other.shape()[0],
            "mat_mul_tn: self 的行数 {} 必须等于 other 的行数 {}（self 以转置参与）",
            self.shape()[0],
            other.shape()[0]
        );
        let self_data = self
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let other_data = other
            .data
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap();
        let result_data = self_data.t().dot(&other_data);
        Self {
            data: into_standard_dyn(result_data).into_shared(),
            source_id: next_source_id(),
        }
    }

    /// 3D 批量矩阵乘法 `[B, m, k] @ [B, k, n] → [B, m, n]`（NN 形式）
    ///
    /// 类似 PyTorch 的 `torch.bmm`：两个操作数必须都是 3D 且 batch 维严格相等，
    /// **不做跨 batch 隐式广播**（符合本项目显式 broadcast 约定）。
    /// 逐 batch 切 2D 视图走与 [`mat_mul`](Self::mat_mul) 相同的 GEMM 路径
    /// （BLAS feature 开启时派发 BLAS），单个 batch 与 2D `mat_mul` 逐 bit 一致。
    pub fn batched_mat_mul(&self, other: &Self) -> Self {
        self.batched_mat_mul_impl(other, false, false)
    }

    /// 3D 批量矩阵乘法 `self @ other^T`（NT 形式，`other` 以转置视图参与）
    ///
    /// 要求 `self: [B, m, k]`、`other: [B, n, k]`，输出 `[B, m, n]`。
    /// 转置只翻转 stride 元数据、零物化拷贝。典型用途：
    /// 批量 MatMul 反向 `dL/dA = upstream @ B^T`。
    pub fn batched_mat_mul_nt(&self, other: &Self) -> Self {
        self.batched_mat_mul_impl(other, false, true)
    }

    /// 3D 批量矩阵乘法 `self^T @ other`（TN 形式，`self` 以转置视图参与）
    ///
    /// 要求 `self: [B, k, m]`、`other: [B, k, n]`，输出 `[B, m, n]`。
    /// 转置零拷贝。典型用途：批量 MatMul 反向 `dL/dB = A^T @ upstream`。
    pub fn batched_mat_mul_tn(&self, other: &Self) -> Self {
        self.batched_mat_mul_impl(other, true, false)
    }

    /// 批量 GEMM 核心：逐 batch 切 2D 视图调 `general_mat_mul` 直写预分配输出
    fn batched_mat_mul_impl(&self, other: &Self, trans_a: bool, trans_b: bool) -> Self {
        use ndarray::{Axis, linalg::general_mat_mul};

        assert!(self.dimension() == 3, "batched_mat_mul: 左操作数必须为 3D");
        assert!(other.dimension() == 3, "batched_mat_mul: 右操作数必须为 3D");

        let a = self
            .data
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap();
        let b = other
            .data
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap();

        let batch = a.dim().0;
        assert!(
            batch == b.dim().0,
            "batched_mat_mul: batch 维必须严格相等（{} vs {}），本框架不做跨 batch 隐式广播",
            batch,
            b.dim().0
        );

        // 逻辑形状（转置视图翻转 m/k、k/n）
        let (m, k_a) = if trans_a {
            (a.dim().2, a.dim().1)
        } else {
            (a.dim().1, a.dim().2)
        };
        let (k_b, n) = if trans_b {
            (b.dim().2, b.dim().1)
        } else {
            (b.dim().1, b.dim().2)
        };
        assert!(
            k_a == k_b,
            "batched_mat_mul: 内维不匹配（A[...,-1]={k_a} vs B[...,-2]={k_b}）"
        );

        let mut out = ndarray::Array3::<f32>::zeros((batch, m, n));
        for i in 0..batch {
            let a_i = a.index_axis(Axis(0), i);
            let b_i = b.index_axis(Axis(0), i);
            let a_i = if trans_a { a_i.reversed_axes() } else { a_i };
            let b_i = if trans_b { b_i.reversed_axes() } else { b_i };
            let mut out_i = out.index_axis_mut(Axis(0), i);
            general_mat_mul(1.0, &a_i, &b_i, 0.0, &mut out_i);
        }

        Self {
            data: out.into_dyn().into_shared(),
            source_id: next_source_id(),
        }
    }
}
