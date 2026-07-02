use crate::tensor::{Tensor, next_source_id};

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
            data: result_data.into_dyn(),
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
            data: result_data.into_dyn(),
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
            data: result_data.into_dyn(),
            source_id: next_source_id(),
        }
    }
}
