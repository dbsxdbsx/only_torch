/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 权重 I/O（save_weights/load_weights）
 *
 * 仅保存/加载注册的参数权重（.bin 格式），
 * 拓扑由用户代码定义，加载前需先构建相同结构的图。
 */

use super::super::error::GraphError;
use super::GraphInner;
use std::path::Path;

impl GraphInner {
    /// 保存模型权重
    ///
    /// 将所有注册的参数保存到二进制文件。
    /// `path` 不含文件后缀，自动添加 `.bin`。
    ///
    /// # 示例
    /// ```ignore
    /// graph.inner().save_weights("models/mnist")?;
    /// // 生成：models/mnist.bin
    /// ```
    pub fn save_weights<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let bin_path = path.with_extension("bin");
        self.save_params(&bin_path)
    }

    /// 加载模型权重
    ///
    /// 从二进制文件加载参数到已构建的图中。
    /// 加载前需先用代码构建相同结构的图。
    ///
    /// # 示例
    /// ```ignore
    /// // 1. 用代码构建图结构（与保存时相同）
    /// let graph = build_mnist_model();
    ///
    /// // 2. 加载权重
    /// graph.inner_mut().load_weights("models/mnist")?;
    /// ```
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let bin_path = path.with_extension("bin");
        self.load_params(&bin_path)
    }
}
