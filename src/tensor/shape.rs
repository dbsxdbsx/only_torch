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
    /// 当 `new_dim` 为 `true` 时，确保所有张量具有相同的形状。除非所有张量都是标量，则它们将堆叠为形状为 `[tensors.len(), 1]` 的张量。
    /// 当 `new_dim` 为 `false`，确保所每个张量的第一个维度可以不同，但其余维度应相同。除非所有张量都是标量，则它们将堆叠为形状为 `[tensors.len()]` 的张量。
    /// 其余情况返回None。
    pub fn stack(tensors: &[&Tensor], new_dim: bool) -> Result<Tensor, &'static str> {
        if tensors.is_empty() {
            return Err("张量列表为空");
        }

        let all_scalars = tensors.iter().all(|t| t.is_scalar());
        let first_shape = tensors[0].shape();

        let compatible_shapes = |t: &Tensor| {
            if all_scalars {
                true
            } else {
                let t_shape = t.shape();
                let skip = if new_dim { 0 } else { 1 };
                t_shape.len() == first_shape.len()
                    && t_shape
                        .iter()
                        .skip(skip)
                        .zip(first_shape.iter().skip(skip))
                        .all(|(a, b)| a == b)
            }
        };

        if !tensors.iter().all(|t| compatible_shapes(t)) {
            return Err("张量形状不兼容");
        }

        let data = tensors
            .iter()
            .flat_map(|t| t.data.as_slice().unwrap())
            .cloned()
            .collect::<Vec<_>>();

        let shape = match (new_dim, all_scalars) {
            (true, true) => vec![tensors.len(), 1],
            (true, false) => {
                let mut shape = tensors[0].shape().to_vec();
                shape.insert(0, tensors.len());
                shape
            }
            (false, true) => vec![tensors.len()],
            (false, false) => {
                let mut shape = tensors[0].shape().to_vec();
                shape[0] = tensors.iter().map(|t| t.shape()[0]).sum();
                shape
            }
        };

        Ok(Tensor::new(&data, &shape))
    }

    pub fn squeeze(&self) -> Tensor {
        let mut new_shape = Vec::new();
        for dim in self.data.shape() {
            if *dim > 1 {
                new_shape.push(*dim);
            }
        }
        let squeezed_data = self.data.clone().into_shape(new_shape).unwrap();
        Tensor {
            data: squeezed_data,
        }
    }

    pub fn squeeze_mut(&mut self) {
        let mut new_shape = Vec::new();
        for dim in self.data.shape() {
            if *dim > 1 {
                new_shape.push(*dim);
            }
        }
        self.data = self
            .data
            .view_mut()
            .into_shape(new_shape)
            .unwrap()
            .to_owned();
    }
}
