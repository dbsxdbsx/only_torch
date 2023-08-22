use super::Tensor;

impl Tensor {
    // TODO：reshape,如将单位矩阵reshape为向量，或者reshape到高阶张量

    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 判断两个张量的形状是否严格一致。如：形状为 [1, 4]，[1, 4]和[4]是不一致的，会返回false
    pub fn is_same_shape(&self, other: &Tensor) -> bool {
        self.shape() == other.shape()
    }

    /// 判断张量是否为标量
    pub fn is_scalar(&self) -> bool {
        self.shape().is_empty() || self.shape().iter().all(|x| *x == 1)
    }

    /// 转化为纯数（number）。若为标量，则返回Some(number)，否则返回None
    pub fn to_number(&self) -> Option<f32> {
        if self.is_scalar() {
            let shape = self.shape();
            let index_array = self.generate_index_array(shape);
            Some(self.data[&index_array[..]])
        } else {
            None
        }
    }
}

// 私有方法
impl Tensor {
    fn generate_index_array(&self, shape: &[usize]) -> Vec<usize> {
        shape.iter().map(|_| 0).collect()
    }
}
