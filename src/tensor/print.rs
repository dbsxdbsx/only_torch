use crate::tensor::Tensor;
use ndarray::{Array, IxDyn};
use std::fmt;

impl Tensor {
    pub fn print(&self) {
        println!("{self}");
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fn display_recursive(
            f: &mut fmt::Formatter,
            data: &Array<f32, IxDyn>,
            indices: &mut Vec<usize>,
            depth: usize,
            rank: usize,
        ) -> fmt::Result {
            if depth == rank {
                write!(f, "{:8.4}", data[&indices[..]])?;
            } else {
                write!(f, "[")?;
                for i in 0..data.shape()[depth] {
                    indices[depth] = i;
                    display_recursive(f, data, indices, depth + 1, rank)?;

                    if i != data.shape()[depth] - 1 {
                        write!(f, ", ")?;
                        if depth == 0 {
                            write!(f, "\n ")?;
                        }
                    }
                }
                write!(f, "]")?;
            }
            Ok(())
        }

        let shape = self.shape();
        let rank = shape.len();
        let mut indices = vec![0; rank];
        if rank > 2 && !self.is_scalar() {
            writeln!(
                f,
                "<对于阶数大于二（rank>2）的张量（形状：{shape:?}）无法展示具体数据>"
            )
        } else {
            display_recursive(f, &self.data, &mut indices, 0, rank)?;
            writeln!(f, "\n形状: {shape:?}")
        }
    }
}
