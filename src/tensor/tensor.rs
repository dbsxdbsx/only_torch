use nalgebra as na;
use rand::distributions::{Distribution, Uniform};

// TODO： 将其拓展到最多3（4?）个维度（因为图像需要3个维度,再加上批处理就有4个维度了）
#[derive(Debug, Clone)]
pub struct Tensor {
    data: na::DMatrix<f32>,
    // data: Vec<na::DMatrix<f32>>,
    shape: Vec<usize>, // 用于存储Tensor的形状。其至少有2个元素，代表行与列，[1，1]表示含还有1个元素的张量（标量）；[1,3]表示1行3列的张量（向量）；[2, 3]表示2行3列的张量（矩阵）
}

impl Tensor {
    pub fn data(&self) -> &na::DMatrix<f32> {
        &self.data
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn zero(rows: usize, cols: usize) -> Self {
        Tensor {
            data: na::DMatrix::zeros(rows, cols),
            shape: vec![rows, cols],
        }
    }

    pub fn scalar(value: f32) -> Self {
        Tensor {
            data: na::DMatrix::from_element(1, 1, value),
            shape: vec![1, 1],
        }
    }

    pub fn to_scalar(&self) -> Option<f32> {
        if self.shape == [1, 1] {
            Some(self.data[(0, 0)])
        } else {
            None
        }
    }

    pub fn random(rows: usize, cols: usize, min: f32, max: f32) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(min, max);
        Tensor {
            data: na::DMatrix::from_fn(rows, cols, |_, _| uniform.sample(&mut rng)),
            shape: vec![rows, cols],
        }
    }

    pub fn eye(size: usize) -> Self {
        Tensor {
            data: na::DMatrix::identity(size, size),
            shape: vec![size, size],
        }
    }
}

//↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓trait点乘↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
// TODO：应该实现点乘的trait，tensor和tensor之间的点乘，tensor和scalar之间的点乘均返回tensor
// 1.tensor和scalar之间的点乘是将scalar广播到tensor的每个元素上；
// 2.tensor和tensor之间的点乘需要保证2者的形状相同；若其中一个形状是[1,1]（即scalar），则遵循1中的原则。否则panic
// 3.不知后期Tensor添加`grad`字段后1和2是否需要修改；
use std::ops::Mul;

impl Mul<f32> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Tensor {
        Tensor {
            data: self.data.map(|x| x * rhs),
            shape: self.shape.clone(),
        }
    }
}

impl Mul<Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        rhs * self
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        if rhs.shape() == &vec![1, 1] {
            let scalar = rhs.to_scalar().unwrap();
            Tensor {
                data: self.data.map(|x| x * scalar),
                shape: self.shape.clone(),
            }
        } else {
            panic!("Unsupported tensor shape for multiplication");
        }
    }
}
//↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑trait点乘↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
use std::fmt;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "形状: {:?}", self.shape())?;

        let max_rows = if self.shape()[0] > 6 {
            3
        } else {
            self.shape()[0]
        };
        let max_cols = if self.shape()[1] > 6 {
            3
        } else {
            self.shape()[1]
        };
        for i in 0..max_rows {
            for j in 0..max_cols {
                write!(f, "{:8.4} ", self.data().index((i, j)))?;
            }
            if self.shape()[1] > 6 {
                write!(f, "   ..  ")?;
                for j in self.shape()[1] - 3..self.shape()[1] {
                    write!(f, "{:8.4} ", self.data().index((i, j)))?;
                }
            }
            writeln!(f)?;
        }
        if self.shape()[0] > 6 {
            let padding = (max_cols * 9) / 2 - 1;
            let padding_str = " ".repeat(padding);
            writeln!(f, "{} ..  ", padding_str)?;
            for i in self.shape()[0] - 3..self.shape()[0] {
                for j in 0..max_cols {
                    write!(f, "{:8.4} ", self.data().index((i, j)))?;
                }
                if self.shape()[1] > 6 {
                    write!(f, "   ..  ")?;
                    for j in self.shape()[1] - 3..self.shape()[1] {
                        write!(f, "{:8.4} ", self.data().index((i, j)))?;
                    }
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
