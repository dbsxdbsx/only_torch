//! # Only Torch
//!
//! `only_torch`项目旨在用纯rust将[pytorch](https://pytorch.org)这类基于梯度的机器学习算法和[NEAT](https://ieeexplore.ieee.org/document/6790655)这类网络突变（类似遗传算法）整合在一起，
//! 打造一个相对来说轻便的跨平台（windows，linux，android...）快速推理AI框架。
//!

pub mod utils;
pub mod logic;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = utils::test_utils(2, 1);
        assert_eq!(result, 4);
    }
}
