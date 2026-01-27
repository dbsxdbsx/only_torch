/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 高层模型 I/O（save_model/load_model）
 *
 * 职责：完整模型的保存/加载（拓扑 JSON + 参数 bin）
 * 依赖：describe() + save_params()/load_params()
 *
 * 与 serialization.rs 的区别：
 * - serialization.rs：底层二进制序列化（只处理参数的原始读写）
 * - model_io.rs：高层模型 I/O（生成/解析 GraphDescriptor + 调用底层序列化）
 */

use super::super::error::GraphError;
use super::GraphInner;
use crate::nn::descriptor::GraphDescriptor;
use std::path::Path;

impl GraphInner {
    /// 保存完整模型（拓扑 JSON + 参数 bin）
    ///
    /// 自动生成两个文件：
    /// - `{path}.json`: 图的拓扑描述（可读）
    /// - `{path}.bin`: 参数数据（紧凑）
    ///
    /// # 示例
    /// ```ignore
    /// graph.save_model("models/mnist")?;
    /// // 生成：models/mnist.json + models/mnist.bin
    /// ```
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let json_path = path.with_extension("json");
        let bin_path = path.with_extension("bin");

        // 1. 保存参数到 bin 文件
        self.save_params(&bin_path)?;

        // 2. 生成描述符并设置 params_file
        let mut descriptor = self.describe();
        descriptor.params_file = Some(bin_path.file_name().map_or_else(
            || "params.bin".to_string(),
            |s| s.to_string_lossy().to_string(),
        ));

        // 3. 保存 JSON
        let json = descriptor
            .to_json()
            .map_err(|e| GraphError::ComputationError(format!("序列化图描述失败: {e}")))?;
        std::fs::write(&json_path, json)
            .map_err(|e| GraphError::ComputationError(format!("写入 JSON 文件失败: {e}")))?;

        Ok(())
    }

    /// 加载模型参数（需要先用代码构建相同结构的图）
    ///
    /// 注意：当前版本不会从 JSON 重建图结构，只加载参数。
    /// 用户需要先用代码构建与保存时相同结构的图，然后调用此方法加载参数。
    ///
    /// # 示例
    /// ```ignore
    /// // 1. 用代码构建图结构（与保存时相同）
    /// let mut graph = build_mnist_model();
    ///
    /// // 2. 加载参数
    /// graph.load_model("models/mnist")?;
    /// ```
    ///
    /// # TODO
    /// 未来版本将支持从 JSON 完整重建图结构，无需预先用代码构建。
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<(), GraphError> {
        let path = path.as_ref();
        let json_path = path.with_extension("json");

        // 1. 读取并解析 JSON
        let json = std::fs::read_to_string(&json_path)
            .map_err(|e| GraphError::ComputationError(format!("读取 JSON 文件失败: {e}")))?;
        let descriptor = GraphDescriptor::from_json(&json)
            .map_err(|e| GraphError::ComputationError(format!("解析图描述失败: {e}")))?;

        // 2. 确定参数文件路径
        let bin_path = if let Some(ref params_file) = descriptor.params_file {
            path.parent().map_or_else(
                || Path::new(params_file).to_path_buf(),
                |p| p.join(params_file),
            )
        } else {
            path.with_extension("bin")
        };

        // 3. 加载参数
        self.load_params(&bin_path)?;

        Ok(())
    }
}
