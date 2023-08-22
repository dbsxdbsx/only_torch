use super::Tensor;

impl Tensor {
    // TODO：reshape,如将单位矩阵reshape为向量，或者reshape到高阶张量

    /// 若为向量，`shape`可以是[n]、[1,n]、[n,1]；
    /// 若为矩阵，`shape`可以是[n,m]；
    /// 若为更高维度的数组，`shape`可以是[c,n,m,...]。
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// 张量的维度、阶（rank）数
    /// 即`shape()`的元素个数--如：形状为`[]`的标量阶数为0，向量阶数为1，矩阵阶数为2，以此类推
    pub fn dims(&self) -> usize {
        self.data.ndim()
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
    pub fn number(&self) -> Option<f32> {
        if self.is_scalar() {
            let shape = self.shape();
            let index_array = self.generate_index_array(shape);
            Some(self.data[&index_array[..]])
        } else {
            None
        }
    }

    /// 将多个张量沿着首个维度堆叠起来，返回一个新的张量。
    /// * `tensors` - 一个包含多个张量的数组的引用。
    /// * `new_dim` - 布尔值，指示是否增加一个新的维度来堆叠。
    ///
    /// 注：张量型张量即使每个张量的形状不一致，也会按照`[1]`的形状进行堆叠。
    pub fn stack(tensors: &[&Tensor], new_dim: bool) -> Option<Tensor> {
        if tensors.is_empty() {
            return None;
        }

        let shape = if tensors.iter().all(|t| t.is_scalar()) {
            if new_dim {
                vec![tensors.len(), 1]
            } else {
                vec![tensors.len()]
            }
        } else if !tensors.iter().all(|t| t.is_same_shape(tensors[0])) {
            return None;
        } else if new_dim {
            let mut shape = tensors[0].shape().to_vec();
            shape.insert(0, tensors.len());
            shape
        } else {
            let mut shape = tensors[0].shape().to_vec();
            shape[0] *= tensors.len();
            shape
        };

        let data = tensors
            .iter()
            .flat_map(|t| t.data.as_slice().unwrap())
            .cloned()
            .collect::<Vec<_>>();

        Some(Tensor::new(&data, &shape))
    }
}

// 私有方法
impl Tensor {
    fn generate_index_array(&self, shape: &[usize]) -> Vec<usize> {
        shape.iter().map(|_| 0).collect()
    }
}
