//! # 常用接口模块
//!
//! 本模块提供一些常用的操作接口

#[cfg(test)]
mod tests;

pub mod macro_for_unit_test;

// TODO: move traits to utils/traits relatively
pub mod traits {
    pub mod float;
    pub mod image;
}
