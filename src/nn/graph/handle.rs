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

    // 注意：checkpoint() 和 prune_nodes_after() 已在方案 C 中移除
    // 新架构下，节点由 Rc 引用计数自动管理，Var 离开作用域时自动释放，
    // 不再需要手动清理节点。

    // ==================== 可视化 ====================

    // Phase 3: save_visualization() 已移除
    // 新的可视化功能请使用 Var::save_visualization() 或 Var::to_dot()
    // 示例：
    //   let output = model.forward(&input)?;
    //   output.save_visualization("model")?;
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}
