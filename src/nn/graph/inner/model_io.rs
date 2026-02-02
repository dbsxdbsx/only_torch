/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 高层模型 I/O（save_model/load_model）
 *
 * Phase 3 简化说明：
 * - 原有的完整模型保存（拓扑 JSON + 参数 bin）依赖已移除的 describe() 方法
 * - 当前版本简化为只保存/加载参数
 * - 拓扑信息通过代码定义，不再序列化
 */

use super::super::error::GraphError;
use super::GraphInner;
use std::path::Path;

impl GraphInner {
    /// 保存模型参数
    ///
    /// 将所有注册的参数保存到二进制文件。
    ///
    /// # 示例
    /// ```ignore
    /// graph.save_model("models/mnist")?;
    /// // 生成：models/mnist.bin
    /// ```
    ///
    /// # 注意
    /// Phase 3 后不再保存拓扑 JSON，只保存参数。
    /// 加载时需要先用代码构建相同结构的图。
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let bin_path = path.with_extension("bin");
        self.save_params(&bin_path)
    }

    /// 加载模型参数
    ///
    /// 从二进制文件加载参数到已构建的图中。
    /// 用户需要先用代码构建与保存时相同结构的图。
    ///
    /// # 示例
    /// ```ignore
    /// // 1. 用代码构建图结构（与保存时相同）
    /// let graph = build_mnist_model();
    ///
    /// // 2. 加载参数
    /// graph.load_model("models/mnist")?;
    /// ```
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let bin_path = path.with_extension("bin");
        self.load_params(&bin_path)
    }
}
