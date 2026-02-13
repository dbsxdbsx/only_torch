/*
 * @Author       : 老董
 * @Date         : 2023-10-21 03:22:26
 * @LastEditors  : 老董
 * @LastEditTime : 2026-02-13
 * @Description  : 张量杂项操作：类型转换、排序、打乱、gather、soft_update、multinomial
 */

use super::super::next_source_id;
use crate::tensor::Tensor;
use ndarray::{Array, Axis, Dimension, IxDyn, Zip};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::thread_rng;

impl From<f32> for Tensor {
    /// 实现 From<f32> trait 用于将`f32`类型转换为形状为`[1,1]`的张量
    fn from(scalar: f32) -> Self {
        Self::new(&[scalar], &[1, 1])
    }
}

// 为Tensor引用类型实现 Into<Tensor> trait
impl<'a> From<&'a Self> for Tensor {
    fn from(tensor: &'a Self) -> Self {
        tensor.clone()
    }
}

// 为f32引用类型实现 Into<Tensor> trait
impl<'a> From<&'a f32> for Tensor {
    fn from(scalar: &'a f32) -> Self {
        Self::new(&[*scalar], &[1, 1])
    }
}

impl Tensor {
    /// 不改变形状情况下，将张量的元素按从小到大的顺序排列，并将其返回（不影响原张量）
    pub fn order(&self) -> Self {
        let flat_data = self.data.view().into_shape(self.data.len()).unwrap();
        let mut sorted_data = flat_data.as_slice().unwrap().to_owned();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ordered_data = Array::from_shape_vec(self.data.shape(), sorted_data).unwrap();
        Self { data: ordered_data, source_id: next_source_id() }
    }

    /// 不改变形状情况下，将张量的元素按从小到大的顺序排列（影响原张量）
    pub fn order_mut(&mut self) {
        let flat_len = self.data.len();
        let mut flat_data = self.data.view_mut().into_shape(flat_len).unwrap();
        let flat_data_slice = flat_data.as_slice_mut().unwrap();
        flat_data_slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
        self.data = flat_data
            .to_owned()
            .into_shape(self.data.shape().to_owned())
            .unwrap();
    }

    /// 打乱张量中的元素顺序，并将其返回（不影响原张量）
    ///
    /// * `dim` - 可选的维度参数，指定沿哪个维度打乱；若为 None 则打乱所有元素
    pub fn shuffle(&self, dim: Option<usize>) -> Self {
        let mut shuffled_data = self.data.clone();
        let mut rng = thread_rng();

        if let Some(dim) = dim {
            let axis = Axis(dim);
            let mut chunks: Vec<_> = shuffled_data
                .axis_iter(axis)
                .map(|c| c.to_owned())
                .collect();
            chunks.shuffle(&mut rng);
            let mut new_data = Array::zeros(shuffled_data.raw_dim());
            for (i, chunk) in chunks.into_iter().enumerate() {
                let mut slice = new_data.index_axis_mut(axis, i);
                slice.assign(&chunk);
            }
            shuffled_data = new_data;
        } else {
            let mut flat_data = shuffled_data.into_shape(self.data.len()).unwrap();
            flat_data.as_slice_mut().unwrap().shuffle(&mut rng);
            shuffled_data = flat_data.into_shape(self.data.shape()).unwrap();
        }

        Self {
            data: shuffled_data,
            source_id: next_source_id(),
        }
    }

    /// 打乱张量中的元素顺序（影响原张量）
    ///
    /// * `dim` - 可选的维度参数，指定沿哪个维度打乱；若为 None 则打乱所有元素
    pub fn shuffle_mut(&mut self, dim: Option<usize>) {
        let mut rng = thread_rng();

        if let Some(dim) = dim {
            let axis = Axis(dim);
            let mut chunks = self
                .data
                .axis_iter(axis)
                .map(|c| c.to_owned())
                .collect::<Vec<_>>();
            chunks.shuffle(&mut rng);
            for (i, chunk) in chunks.into_iter().enumerate() {
                let mut slice = self.data.index_axis_mut(axis, i);
                slice.assign(&chunk);
            }
        } else {
            let flat_len = self.data.len();
            let mut flat_data = self.data.view_mut().into_shape(flat_len).unwrap();
            let flat_data_slice = flat_data.as_slice_mut().unwrap();
            flat_data_slice.shuffle(&mut rng);
            self.data = flat_data
                .to_owned()
                .into_shape(self.data.shape().to_owned())
                .unwrap();
        }
    }

    /// 使用指定种子打乱张量中的元素顺序（影响原张量，确保可重复性）
    ///
    /// * `dim` - 可选的维度参数，指定沿哪个维度打乱；若为 None 则打乱所有元素
    /// * `seed` - 随机数生成器的种子
    pub fn shuffle_mut_seeded(&mut self, dim: Option<usize>, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        if let Some(dim) = dim {
            let axis = Axis(dim);
            let mut chunks = self
                .data
                .axis_iter(axis)
                .map(|c| c.to_owned())
                .collect::<Vec<_>>();
            chunks.shuffle(&mut rng);
            for (i, chunk) in chunks.into_iter().enumerate() {
                let mut slice = self.data.index_axis_mut(axis, i);
                slice.assign(&chunk);
            }
        } else {
            let flat_len = self.data.len();
            let mut flat_data = self.data.view_mut().into_shape(flat_len).unwrap();
            let flat_data_slice = flat_data.as_slice_mut().unwrap();
            flat_data_slice.shuffle(&mut rng);
            self.data = flat_data
                .to_owned()
                .into_shape(self.data.shape().to_owned())
                .unwrap();
        }
    }

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓gather↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 按索引张量从指定维度收集元素
    ///
    /// 类似 `PyTorch` 的 `torch.gather(input, dim, index)`。
    /// 对于 2D 张量 `input[M, N]` 和索引 `index[M, K]`（dim=1 时）：
    /// - `output[i, j] = input[i, index[i, j]]`
    ///
    /// # 参数
    /// - `dim`: 沿哪个维度进行 gather 操作
    /// - `index`: 索引张量（元素为 f32，会被转换为 usize）
    ///
    /// # 返回
    /// 与 `index` 形状相同的张量
    ///
    /// # Panics
    /// - 如果 `dim` 超出维度范围
    /// - 如果 `index` 和 `self` 的维度数不同
    /// - 如果 `index` 中除 `dim` 维度外的其他维度大小与 `self` 不匹配
    /// - 如果 `index` 中的索引值超出 `self` 在 `dim` 维度的范围
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// // SAC/DQN 场景：按动作索引选择 Q 值
    /// // Q 值：[[1.0, 2.0, 3.0],   (batch=2, action_dim=3)
    /// //        [4.0, 5.0, 6.0]]
    /// let q_values = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    ///
    /// // 动作索引：[[1],   (选第 0 行的索引 1)
    /// //           [2]]   (选第 1 行的索引 2)
    /// let actions = Tensor::new(&[1.0, 2.0], &[2, 1]);
    ///
    /// let selected = q_values.gather(1, &actions);
    /// // selected = [[2.0],   <- q_values[0, 1]
    /// //             [6.0]]   <- q_values[1, 2]
    /// assert_eq!(selected.shape(), &[2, 1]);
    /// assert_eq!(selected[[0, 0]], 2.0);
    /// assert_eq!(selected[[1, 0]], 6.0);
    /// ```
    pub fn gather(&self, dim: usize, index: &Tensor) -> Tensor {
        let self_shape = self.shape();
        let index_shape = index.shape();
        let ndim = self_shape.len();

        // 1. 验证 dim
        assert!(dim < ndim, "gather: dim {} 超出张量维度 {}", dim, ndim);

        // 2. 验证 index 维度数与 self 相同
        assert!(
            index_shape.len() == ndim,
            "gather: index 维度数 {} 必须与输入张量维度数 {} 相同",
            index_shape.len(),
            ndim
        );

        // 3. 验证除 dim 外的其他维度大小一致
        for d in 0..ndim {
            if d != dim {
                assert!(
                    index_shape[d] == self_shape[d],
                    "gather: 维度 {} 上 index 大小 {} 与输入张量大小 {} 不匹配",
                    d,
                    index_shape[d],
                    self_shape[d]
                );
            }
        }

        // 4. 计算输出大小并收集元素
        let output_size: usize = index_shape.iter().product();
        let mut output_data = Vec::with_capacity(output_size);

        // 使用 ndindex 遍历 index 张量的每个位置
        for idx in ndarray::indices(index_shape) {
            // 获取 index 中的值作为 gather 索引
            let gather_idx = index.data[&idx] as usize;

            // 验证索引范围
            assert!(
                gather_idx < self_shape[dim],
                "gather: 索引 {} 超出维度 {} 的范围 [0, {})",
                gather_idx,
                dim,
                self_shape[dim]
            );

            // 构建从 self 中取值的索引
            let mut self_idx: Vec<usize> = idx.as_array_view().to_vec();
            self_idx[dim] = gather_idx;

            // 取值
            output_data.push(self.data[IxDyn(&self_idx)]);
        }

        Tensor::new(&output_data, index_shape)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑gather↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓soft_update↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 参数软更新（Soft Update）
    ///
    /// 用于强化学习中目标网络的平滑更新：
    /// `self = τ × source + (1 - τ) × self`
    ///
    /// # 参数
    /// - `source`: 源张量（如在线网络参数）
    /// - `tau`: 更新系数，通常取较小值（如 0.005）
    ///
    /// # Panics
    /// 如果 `self` 和 `source` 形状不匹配
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let mut target = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let online = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    ///
    /// target.soft_update(&online, 0.1);
    /// // target = 0.1 * [10, 20, 30, 40] + 0.9 * [1, 2, 3, 4]
    /// //        = [1.9, 3.8, 5.7, 7.6]
    /// ```
    pub fn soft_update(&mut self, source: &Tensor, tau: f32) {
        assert_eq!(
            self.shape(),
            source.shape(),
            "soft_update: 形状不匹配，self: {:?}, source: {:?}",
            self.shape(),
            source.shape()
        );

        // self = τ * source + (1 - τ) * self
        Zip::from(&mut self.data)
            .and(&source.data)
            .for_each(|target, &src| {
                *target = tau * src + (1.0 - tau) * *target;
            });
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑soft_update↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/

    /*↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓multinomial↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓*/
    /// 按概率向量进行多项分布采样
    ///
    /// 类似 PyTorch 的 `torch.multinomial(probs, num_samples)`。
    /// 对每一行概率向量独立采样，返回采样得到的类别索引。
    ///
    /// # 参数
    /// - `num_samples` — 每行采样次数
    ///
    /// # 输入
    /// 2D 张量 `[batch, num_classes]`，每行为归一化概率（和为 1）
    ///
    /// # 返回
    /// 2D 张量 `[batch, num_samples]`，元素为采样到的类别索引（f32）
    ///
    /// # 示例
    /// ```
    /// use only_torch::tensor::Tensor;
    ///
    /// let probs = Tensor::new(&[0.1, 0.7, 0.2], &[1, 3]);
    /// let sample = probs.multinomial(1);
    /// assert_eq!(sample.shape(), &[1, 1]);
    /// // sample[[0, 0]] 为 0, 1, 或 2（按概率 0.1, 0.7, 0.2 采样）
    /// ```
    pub fn multinomial(&self, num_samples: usize) -> Self {
        let mut rng = thread_rng();
        self.multinomial_with_rng(num_samples, &mut rng)
    }

    /// 带 RNG 的多项分布采样（可复现）
    ///
    /// 与 `multinomial()` 功能相同，但使用指定的 RNG 以确保可复现性。
    pub fn multinomial_with_rng(
        &self,
        num_samples: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let shape = self.shape();
        assert_eq!(
            shape.len(),
            2,
            "multinomial: 输入须为 2D [batch, num_classes]，实际维度: {}",
            shape.len()
        );
        assert!(
            num_samples >= 1,
            "multinomial: num_samples 至少为 1"
        );

        let batch = shape[0];
        let num_classes = shape[1];
        let mut result = Vec::with_capacity(batch * num_samples);

        for b in 0..batch {
            for _ in 0..num_samples {
                let r: f32 = rng.gen_range(0.0..1.0);
                let mut cumsum = 0.0;
                let mut sampled = num_classes - 1; // 兜底：概率和不足时取最后一类
                for c in 0..num_classes {
                    cumsum += self[[b, c]];
                    if r < cumsum {
                        sampled = c;
                        break;
                    }
                }
                result.push(sampled as f32);
            }
        }

        Self::new(&result, &[batch, num_samples])
    }

    /// 带种子的多项分布采样（可复现）
    pub fn multinomial_seeded(&self, num_samples: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        self.multinomial_with_rng(num_samples, &mut rng)
    }
    /*↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑multinomial↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑*/
}
