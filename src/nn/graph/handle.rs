/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : Graph 句柄（用户级 API）
 */

use super::error::GraphError;
use super::inner::GraphInner;
use crate::nn::var::{Init, Var};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

/// Graph - 计算图句柄（PyTorch 风格用户 API）
///
/// # 设计原则
/// - 是 `Rc<RefCell<GraphInner>>` 的薄封装
/// - Clone 语义：多个 Graph 引用同一个 `GraphInner`
/// - 创建的 Var 自动持有图引用
#[derive(Clone)]
pub struct Graph {
    inner: Rc<RefCell<GraphInner>>,
    /// 模型名称（可选，用于可视化分组）
    ///
    /// 通过 `with_model_name("Generator")` 设置后，该 Graph 创建的所有 Layer
    /// 会自动将层名拼接为 `"Generator/fc1"` 格式，可视化时渲染为嵌套 cluster。
    /// 不设置则无模型分组，行为与默认一致。
    model_name: Option<String>,
}

impl Graph {
    // ==================== 创建 ====================

    /// 创建新图
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::new())),
            model_name: None,
        }
    }

    /// 创建带种子的图（用于确定性训练）
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::new_with_seed(seed))),
            model_name: None,
        }
    }

    /// 从现有 `GraphInner` 创建句柄
    pub fn from_inner(inner: GraphInner) -> Self {
        Self {
            inner: Rc::new(RefCell::new(inner)),
            model_name: None,
        }
    }

    /// 从现有 Rc 创建句柄
    pub(crate) fn from_rc(inner: Rc<RefCell<GraphInner>>) -> Self {
        Self {
            inner,
            model_name: None,
        }
    }

    // ==================== 模型分组 ====================

    /// 创建带模型名的 Graph（用于可视化分组）
    ///
    /// 返回一个新的 Graph clone，共享同一个 `GraphInner`，
    /// 但携带模型名。用这个 Graph 创建的 Layer 会自动将层名
    /// 拼接为 `"模型名/层名"` 格式，可视化时渲染为嵌套 cluster。
    ///
    /// # 示例
    /// ```ignore
    /// let graph = graph.with_model_name("Generator");
    /// // 后续创建的层名自动变为 "Generator/fc1"、"Generator/fc2"
    /// let fc1 = Linear::new(&graph, 64, 128, true, "fc1")?;
    /// ```
    pub fn with_model_name(&self, name: &str) -> Self {
        Self {
            inner: Rc::clone(&self.inner),
            model_name: Some(name.to_string()),
        }
    }

    /// 获取当前模型名（供 Layer 查询）
    pub fn model_name(&self) -> Option<&str> {
        self.model_name.as_deref()
    }

    /// 获取内部 `GraphInner` 的不可变引用
    pub fn inner(&self) -> std::cell::Ref<'_, GraphInner> {
        self.inner.borrow()
    }

    /// 获取内部 `GraphInner` 的可变引用
    pub fn inner_mut(&self) -> std::cell::RefMut<'_, GraphInner> {
        self.inner.borrow_mut()
    }

    /// 获取内部 Rc
    pub(crate) fn inner_rc(&self) -> Rc<RefCell<GraphInner>> {
        Rc::clone(&self.inner)
    }

    /// 获取注册的参数数量
    ///
    /// 返回通过 `register_parameter()` 注册且仍存活的参数数量。
    /// Phase 3 后不再跟踪所有节点，只跟踪参数。
    pub fn parameter_count(&self) -> usize {
        self.inner.borrow().get_all_parameters().len()
    }

    // ==================== 创建变量 ====================

    /// 创建输入节点并设置数据
    pub fn input(&self, data: &Tensor) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_basic_input_node(data.shape(), None)?;
        node.set_value(Some(data))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建命名输入节点
    pub fn input_named(&self, data: &Tensor, name: &str) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_basic_input_node(data.shape(), Some(name))?;
        node.set_value(Some(data))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建带形状的输入节点
    pub fn input_shape(&self, shape: &[usize], name: Option<&str>) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_basic_input_node(shape, name)?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建参数节点
    pub fn parameter(&self, shape: &[usize], init: Init, name: &str) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let init_data = if let Some(ref mut rng) = g.rng {
            init.generate_with_rng(shape, rng)
        } else {
            init.generate(shape)
        };
        let node = g.create_parameter_node(shape, Some(name))?;
        // 注册参数到 GraphInner（使 zero_grad/parameter_count 等正常工作）
        g.register_parameter(name.to_string(), std::rc::Rc::downgrade(&node))?;
        drop(g); // 释放借用
        node.set_value(Some(&init_data))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建零张量
    pub fn zeros(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_basic_input_node(shape, None)?;
        node.set_value(Some(&Tensor::zeros(shape)))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建 Target 输入节点
    pub fn target(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_target_input_node(shape, None)?;
        node.set_value(Some(&Tensor::zeros(shape)))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建全一张量
    pub fn ones(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_basic_input_node(shape, None)?;
        node.set_value(Some(&Tensor::ones(shape)))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建随机张量
    pub fn randn(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_basic_input_node(shape, None)?;
        let data = Tensor::normal(0.0, 1.0, shape);
        node.set_value(Some(&data))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建动态零张量节点
    ///
    /// 以 `reference` 为父节点，前向传播时自动读取其 batch_size 生成零张量。
    /// 反向传播时不向 reference 传播梯度（`ZerosLike` 是常量节点）。
    pub fn zeros_like(
        &self,
        reference: &Var,
        feature_shape: &[usize],
        name: Option<&str>,
    ) -> Result<Var, GraphError> {
        let node = self.inner.borrow_mut().create_zeros_like_node(
            std::rc::Rc::clone(reference.node()),
            feature_shape,
            name,
        )?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建常量张量
    pub fn constant(&self, data: &Tensor) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_basic_input_node(data.shape(), None)?;
        node.set_value(Some(data))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    /// 创建命名常量张量
    pub fn constant_named(&self, data: &Tensor, name: &str) -> Result<Var, GraphError> {
        let node = self
            .inner
            .borrow_mut()
            .create_basic_input_node(data.shape(), Some(name))?;
        node.set_value(Some(data))?;
        Ok(Var::new_with_rc_graph(node, &self.inner))
    }

    // ==================== 执行 ====================

    /// 前向传播
    pub fn forward(&self, output: &Var) -> Result<(), GraphError> {
        self.inner
            .borrow_mut()
            .forward_via_node_inner(output.node())
    }

    /// 反向传播
    pub fn backward(&self, loss: &Var) -> Result<f32, GraphError> {
        loss.backward()
    }

    // ==================== 训练控制 ====================

    /// 清零所有参数的梯度
    pub fn zero_grad(&self) -> Result<(), GraphError> {
        self.inner.borrow_mut().zero_grad()
    }

    /// 设置训练模式
    pub fn train(&self) {
        self.inner.borrow_mut().set_train_mode();
    }

    /// 设置评估模式
    pub fn eval(&self) {
        self.inner.borrow_mut().set_eval_mode();
    }

    /// 是否处于评估模式
    pub fn is_eval(&self) -> bool {
        !self.inner.borrow().is_train_mode()
    }

    /// 在 `no_grad` 上下文中执行闭包
    pub fn no_grad_scope<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&Self) -> R,
    {
        let was_train = !self.is_eval();
        self.eval();
        let result = f(self);
        if was_train {
            self.train();
        }
        result
    }

    // 注意：checkpoint() 和 prune_nodes_after() 已移除
    // 新架构下，节点由 Rc 引用计数自动管理，Var 离开作用域时自动释放，
    // 不再需要手动清理节点。

    // ==================== 可视化快照 ====================

    /// 快照当前计算图拓扑（仅首次调用生效，后续自动跳过）
    ///
    /// 在训练循环中 Var 还活着时调用，快照后 Var 可安全 drop。
    /// 之后任意时刻调用 `visualize_snapshot` 生成可视化文件。
    ///
    /// # 参数
    /// - `named_outputs`: 命名输出端点，每个 `(名称, &Var)` 对应一条优化路径
    ///
    /// # 示例
    /// ```ignore
    /// // 训练步骤末尾，backward 之后
    /// graph.snapshot_once(&[
    ///     ("Actor Loss", &actor_loss),
    ///     ("Critic Loss", &critic1_loss),
    /// ]);
    /// ```
    pub fn snapshot_once(&self, named_outputs: &[(&str, &Var)]) {
        let mut inner = self.inner.borrow_mut();
        if inner.visualization_snapshot.is_some() {
            return; // 已拍过快照，跳过
        }
        let snapshot = Var::build_snapshot(named_outputs);
        inner.visualization_snapshot = Some(snapshot);
    }

    /// 快照当前计算图拓扑（无名称版本，自动命名为 "Output 1", "Output 2"...）
    ///
    /// 功能同 `snapshot_once`，适用于不需要路径名称的场景。
    ///
    /// # 示例
    /// ```ignore
    /// graph.snapshot_once_from(&[&loss]);
    /// ```
    pub fn snapshot_once_from(&self, outputs: &[&Var]) {
        let mut inner = self.inner.borrow_mut();
        if inner.visualization_snapshot.is_some() {
            return;
        }
        let named: Vec<(String, &Var)> = outputs
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let name = if outputs.len() == 1 {
                    "Output".to_string()
                } else {
                    format!("Output {}", i + 1)
                };
                (name, *v)
            })
            .collect();
        let named_refs: Vec<(&str, &Var)> = named.iter().map(|(n, v)| (n.as_str(), *v)).collect();
        let snapshot = Var::build_snapshot(&named_refs);
        inner.visualization_snapshot = Some(snapshot);
    }

    /// 从已存储的快照渲染可视化文件（.dot + .png）
    ///
    /// 必须先调用 `snapshot_once` 或 `snapshot_once_from`。
    /// 可在训练结束后任意时刻调用，不依赖 Var 生命周期。
    ///
    /// # 示例
    /// ```ignore
    /// graph.visualize_snapshot("examples/cartpole_sac/cartpole_sac")?;
    /// ```
    pub fn visualize_snapshot<P: AsRef<std::path::Path>>(
        &self,
        base_path: P,
    ) -> Result<super::VisualizationOutput, GraphError> {
        use std::fs::File;
        use std::io::Write;
        use std::process::Command;

        let inner = self.inner.borrow();
        let snapshot = inner.visualization_snapshot.as_ref().ok_or_else(|| {
            GraphError::InvalidOperation(
                "未调用 snapshot_once，无法渲染可视化。请在训练循环中先调用 graph.snapshot_once(...)".to_string(),
            )
        })?;

        let base = base_path.as_ref();
        if let Some(ext) = base.extension() {
            return Err(GraphError::InvalidOperation(format!(
                "base_path 不应包含文件后缀（如 .{}），请使用不带后缀的路径",
                ext.to_string_lossy()
            )));
        }

        let dot_path = base.with_extension("dot");
        let png_path = base.with_extension("png");

        // 从快照 + 图元数据生成 DOT
        let layer_groups = inner.layer_groups().to_vec();
        let recurrent_metas = inner.recurrent_layer_metas().to_vec();
        let dot_content = Var::snapshot_to_dot(snapshot, &layer_groups, &recurrent_metas);

        // 保存 .dot 文件
        {
            let mut file = File::create(&dot_path)
                .map_err(|e| GraphError::ComputationError(format!("无法创建 DOT 文件: {}", e)))?;
            file.write_all(dot_content.as_bytes())
                .map_err(|e| GraphError::ComputationError(format!("写入 DOT 文件失败: {}", e)))?;
            file.sync_all()
                .map_err(|e| GraphError::ComputationError(format!("同步 DOT 文件失败: {}", e)))?;
        }

        // 尝试用 Graphviz 生成 PNG
        let graphviz_available = Command::new("dot")
            .arg("-V")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);

        if graphviz_available {
            let output = Command::new("dot")
                .arg("-Tpng")
                .arg(&dot_path)
                .arg("-o")
                .arg(&png_path)
                .output();

            match output {
                Ok(o) if o.status.success() => Ok(super::VisualizationOutput {
                    dot_path,
                    image_path: Some(png_path),
                    graphviz_available: true,
                    graphviz_hint: None,
                }),
                Ok(o) => {
                    let stderr = String::from_utf8_lossy(&o.stderr);
                    let hint = if stderr.is_empty() {
                        format!("Graphviz 执行失败 (exit: {:?})", o.status.code())
                    } else {
                        format!(
                            "Graphviz 执行失败 (exit: {:?}): {}",
                            o.status.code(),
                            stderr.trim()
                        )
                    };
                    Ok(super::VisualizationOutput {
                        dot_path,
                        image_path: None,
                        graphviz_available: true,
                        graphviz_hint: Some(hint),
                    })
                }
                Err(e) => Ok(super::VisualizationOutput {
                    dot_path,
                    image_path: None,
                    graphviz_available: true,
                    graphviz_hint: Some(format!("无法执行 Graphviz: {}", e)),
                }),
            }
        } else {
            Ok(super::VisualizationOutput {
                dot_path,
                image_path: None,
                graphviz_available: false,
                graphviz_hint: Some("请安装 Graphviz: https://graphviz.org/download/".to_string()),
            })
        }
    }

    /// 清除已有的可视化快照（允许重新拍摄）
    pub fn clear_snapshot(&self) {
        self.inner.borrow_mut().visualization_snapshot = None;
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
