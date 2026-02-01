/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 节点构建方法（new_*_node）
 *
 * 方案 C 新增：
 * - create_node_inner(): 通用节点创建，返回 Rc<NodeInner>
 * - create_basic_input_node() 等: 各节点类型的新创建方法
 */

use super::super::error::GraphError;
use super::GraphInner;
use crate::nn::NodeId;
use crate::nn::nodes::raw_node::Reduction;
use crate::nn::nodes::{NodeHandle, NodeInner, NodeType};
use std::rc::Rc;

impl GraphInner {
    // ==================== 方案 C：新节点创建 API ====================

    /// 创建 NodeInner（方案 C 核心方法）
    ///
    /// # 参数
    /// - `raw_node`: 节点类型（NodeType）
    /// - `name`: 可选的节点名称
    /// - `node_type_str`: 节点类型字符串（用于生成默认名称）
    /// - `parents`: 父节点的强引用列表
    ///
    /// # 过渡期行为
    /// - 设置 forward_edges/backward_edges（保持与旧代码兼容）
    ///
    /// # 返回
    /// 返回 `Rc<NodeInner>`，调用者持有强引用
    pub(in crate::nn::graph) fn create_node_inner(
        &mut self,
        raw_node: NodeType,
        name: Option<&str>,
        node_type_str: &str,
        parents: Vec<Rc<NodeInner>>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        // 生成 ID 和名称
        let node_id = self.generate_valid_node_id();
        let node_name = self.generate_valid_new_node_name(name.unwrap_or(""), node_type_str)?;

        // 创建 NodeInner
        let node_inner = Rc::new(NodeInner::new(
            node_id,
            Some(node_name),
            raw_node,
            parents.clone(),
        ));

        // 过渡期：设置边（供旧代码的可视化等功能使用）
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();
        for &parent_id in &parent_ids {
            self.forward_edges
                .entry(parent_id)
                .or_default()
                .push(node_id);
        }
        self.backward_edges
            .entry(node_id)
            .or_default()
            .extend(&parent_ids);

        Ok(node_inner)
    }

    /// 创建 BasicInput 节点（方案 C 新 API）
    ///
    /// 返回 `Rc<NodeInner>`，这是一个叶子节点（无父节点）
    pub fn create_basic_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::InputVariant;

        let input_variant = InputVariant::new_data(shape)?;
        let raw_node: NodeType = input_variant.into();

        self.create_node_inner(raw_node, name, "input", vec![])
    }

    /// 创建 TargetInput 节点（方案 C 新 API）
    ///
    /// 用于 Loss 的目标值（真实标签），支持动态 batch。
    /// 返回 `Rc<NodeInner>`，这是一个叶子节点（无父节点）
    pub fn create_target_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::InputVariant;

        let input_variant = InputVariant::new_target(shape);
        let raw_node: NodeType = input_variant.into();

        self.create_node_inner(raw_node, name, "target", vec![])
    }

    /// 创建 SmartInput 节点（方案 C 新 API）
    ///
    /// 用于 ModelState 的智能入口，支持动态 batch 和梯度路由。
    /// 返回 `Rc<NodeInner>`，这是一个叶子节点（无父节点）
    pub fn create_smart_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::InputVariant;

        let input_variant = InputVariant::new_smart(shape);
        let raw_node: NodeType = input_variant.into();

        self.create_node_inner(raw_node, name, "input", vec![])
    }

    /// 创建 RecurrentOutput 节点（方案 C 新 API）
    ///
    /// 用于 RNN/LSTM/GRU 循环层的输出桥接。
    /// 返回 `Rc<NodeInner>`，这是一个叶子节点（无父节点）
    pub fn create_recurrent_output_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::InputVariant;

        let input_variant = InputVariant::new_recurrent_output(shape);
        let raw_node: NodeType = input_variant.into();

        self.create_node_inner(raw_node, name, "recurrent_output", vec![])
    }

    /// 创建 Parameter 节点（方案 C 新 API）
    ///
    /// 可训练参数（权重、偏置等），支持 Kaiming 初始化。
    /// 如果 Graph 有固定种子，使用该种子初始化；否则使用随机初始化。
    /// 返回 `Rc<NodeInner>`，这是一个叶子节点（无父节点）
    pub fn create_parameter_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Parameter;

        let param = if let Some(ref mut rng) = self.rng {
            use rand::Rng;
            let seed: u64 = rng.r#gen();
            Parameter::new_seeded(shape, seed)?
        } else {
            Parameter::new(shape)?
        };
        let raw_node: NodeType = param.into();

        self.create_node_inner(raw_node, name, "parameter", vec![])
    }

    /// 创建带固定种子的 Parameter 节点（方案 C 新 API）
    ///
    /// 使用指定种子进行 Kaiming 初始化，确保可重复性。
    pub fn create_parameter_node_seeded(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
        seed: u64,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Parameter;

        let param = Parameter::new_seeded(shape, seed)?;
        let raw_node: NodeType = param.into();

        self.create_node_inner(raw_node, name, "parameter", vec![])
    }

    /// 创建 State 节点（方案 C 新 API）
    ///
    /// 用于 RNN 中的时间状态（隐藏状态 h、LSTM 的 c 等）。
    /// 支持动态 batch，第一维可以是任意值。
    /// 返回 `Rc<NodeInner>`，这是一个叶子节点（无父节点）
    pub fn create_state_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::State;

        let state = State::new(shape)?;
        let raw_node: NodeType = state.into();

        self.create_node_inner(raw_node, name, "state", vec![])
    }

    /// 创建 Add 节点（方案 C 新 API）
    ///
    /// 逐元素加法，支持广播。
    /// 返回 `Rc<NodeInner>`，父节点引用由 `parents` 参数传入。
    pub fn create_add_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Add;

        // 1. 从父节点提取形状信息
        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();

        // 2. 使用 new_from_shapes 创建 Add 节点
        let add = Add::new_from_shapes(&parent_shapes_ref, &parent_dynamic_shapes)?;
        let raw_node: NodeType = add.into();

        // 3. 创建 NodeInner 并注册
        self.create_node_inner(raw_node, name, "add", parents)
    }

    /// 创建 Subtract 节点（方案 C 新 API）
    ///
    /// 逐元素减法 (left - right)，支持广播。
    /// 返回 `Rc<NodeInner>`，父节点引用由 `parents` 参数传入。
    pub fn create_subtract_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Subtract;

        // 1. 从父节点提取形状信息
        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        // 2. 使用 new_from_shapes 创建 Subtract 节点
        let subtract =
            Subtract::new_from_shapes(&parent_shapes_ref, &parent_dynamic_shapes, parent_ids)?;
        let raw_node: NodeType = subtract.into();

        // 3. 创建 NodeInner 并注册
        self.create_node_inner(raw_node, name, "subtract", parents)
    }

    /// 创建 Multiply 节点（方案 C 新 API）
    ///
    /// 逐元素乘法（Hadamard积），支持广播。
    /// 返回 `Rc<NodeInner>`，父节点引用由 `parents` 参数传入。
    pub fn create_multiply_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Multiply;

        // 1. 从父节点提取形状信息
        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        // 2. 使用 new_from_shapes 创建 Multiply 节点
        let multiply =
            Multiply::new_from_shapes(&parent_shapes_ref, &parent_dynamic_shapes, parent_ids)?;
        let raw_node: NodeType = multiply.into();

        // 3. 创建 NodeInner 并注册
        self.create_node_inner(raw_node, name, "multiply", parents)
    }

    /// 创建 Divide 节点（方案 C 新 API）
    ///
    /// 逐元素除法 (left / right)，支持广播。
    /// 返回 `Rc<NodeInner>`，父节点引用由 `parents` 参数传入。
    pub fn create_divide_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Divide;

        // 1. 从父节点提取形状信息
        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        // 2. 使用 new_from_shapes 创建 Divide 节点
        let divide =
            Divide::new_from_shapes(&parent_shapes_ref, &parent_dynamic_shapes, parent_ids)?;
        let raw_node: NodeType = divide.into();

        // 3. 创建 NodeInner 并注册
        self.create_node_inner(raw_node, name, "divide", parents)
    }

    // ==================== 旧节点创建 API（过渡期保留）====================

    /// 添加节点到列表
    pub(in crate::nn::graph) fn add_node_to_list(
        &mut self,
        mut node_handle: NodeHandle,
        name: Option<&str>,
        node_type: &str,
        parents: &[NodeId],
    ) -> Result<NodeId, GraphError> {
        let node_id = self.generate_valid_node_id();
        let node_name = self.generate_valid_new_node_name(name.unwrap_or(""), node_type)?;

        for &parent_id in parents {
            self.forward_edges
                .entry(parent_id)
                .or_default()
                .push(node_id);
        }
        self.backward_edges
            .entry(node_id)
            .or_default()
            .extend(parents);

        node_handle.bind_id_and_name(node_id, &node_name);
        self.nodes.insert(node_id, node_handle);
        Ok(node_id)
    }

    /// 创建基本输入节点
    pub fn new_basic_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_basic_input(shape)?;
        self.add_node_to_list(node, name, "input", &[])
    }

    /// 创建目标输入节点
    pub fn new_target_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_target_input(shape);
        self.add_node_to_list(node, name, "target", &[])
    }

    /// 创建 `SmartInput` 节点
    pub fn new_smart_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_smart_input(shape)?;
        self.add_node_to_list(node, name, "input", &[])
    }

    /// 创建 `RecurrentOutput` 节点
    pub fn new_recurrent_output_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_recurrent_output(shape)?;
        self.add_node_to_list(node, name, "recurrent_output", &[])
    }

    /// 设置 SmartInput/RecurrentOutput 节点的 detached 状态
    pub fn set_router_detached(
        &mut self,
        node_id: NodeId,
        detached: bool,
        mark_ever_detached: bool,
    ) -> Result<(), GraphError> {
        let node = self.get_node_mut(node_id)?;
        node.set_router_detached(detached, mark_ever_detached)
    }

    /// 设置梯度路由目标
    pub fn set_gradient_target(
        &mut self,
        node_id: NodeId,
        target: Option<NodeId>,
    ) -> Result<(), GraphError> {
        let node = self.get_node_mut(node_id)?;
        node.set_gradient_target(target)
    }

    /// 获取梯度路由目标
    pub fn get_gradient_target(&self, node_id: NodeId) -> Result<Option<NodeId>, GraphError> {
        let node = self.get_node(node_id)?;
        Ok(node.gradient_target())
    }

    /// 创建参数节点
    pub fn new_parameter_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = if let Some(ref mut rng) = self.rng {
            use rand::Rng;
            let seed: u64 = rng.r#gen();
            NodeHandle::new_parameter_seeded(shape, seed)?
        } else {
            NodeHandle::new_parameter(shape)?
        };
        self.add_node_to_list(node, name, "parameter", &[])
    }

    /// 创建带种子的参数节点
    pub fn new_parameter_node_seeded(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
        seed: u64,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_parameter_seeded(shape, seed)?;
        self.add_node_to_list(node, name, "parameter", &[])
    }

    /// 创建 State 节点
    pub fn new_state_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_state(shape)?;
        self.add_node_to_list(node, name, "state", &[])
    }

    /// 创建 `ZerosLike` 节点
    pub fn new_zeros_like_node(
        &mut self,
        reference: NodeId,
        feature_shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_zeros_like(feature_shape);
        self.add_node_to_list(node, name, "zeros_like", &[reference])
    }

    pub fn new_add_node(
        &mut self,
        parents: &[NodeId],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_add(&self.get_nodes(parents)?)?;
        self.add_node_to_list(handle, name, "add", parents)
    }

    /// 创建 Stack 节点（多张量堆叠/拼接）
    ///
    /// # 参数
    /// - `parents`: 要堆叠的父节点 ID 列表
    /// - `axis`: 沿哪个轴进行操作
    /// - `new_dim`: true 表示插入新维度（stack），false 表示沿现有维度拼接（concat）
    /// - `name`: 可选的节点名称
    pub fn new_stack_node(
        &mut self,
        parents: &[NodeId],
        axis: usize,
        new_dim: bool,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_stack(&self.get_nodes(parents)?, axis, new_dim)?;
        // 根据 new_dim 决定节点类型名：stack（新增维度）或 concat（拼接）
        let type_name = if new_dim { "stack" } else { "concat" };
        self.add_node_to_list(handle, name, type_name, parents)
    }

    pub fn new_conv2d_node(
        &mut self,
        input_id: NodeId,
        kernel_id: NodeId,
        stride: (usize, usize),
        padding: (usize, usize),
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_conv2d(&self.get_nodes(&[input_id, kernel_id])?, stride, padding)?;
        self.add_node_to_list(handle, name, "conv2d", &[input_id, kernel_id])
    }

    pub fn new_max_pool2d_node(
        &mut self,
        input_id: NodeId,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_max_pool2d(&self.get_nodes(&[input_id])?, kernel_size, stride)?;
        self.add_node_to_list(handle, name, "max_pool2d", &[input_id])
    }

    pub fn new_avg_pool2d_node(
        &mut self,
        input_id: NodeId,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle =
            NodeHandle::new_avg_pool2d(&self.get_nodes(&[input_id])?, kernel_size, stride)?;
        self.add_node_to_list(handle, name, "avg_pool2d", &[input_id])
    }

    pub fn new_mat_mul_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_mat_mul(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "mat_mul", &[left_node_id, right_node_id])
    }

    pub fn new_multiply_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_multiply(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "multiply", &[left_node_id, right_node_id])
    }

    pub fn new_divide_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_divide(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "divide", &[left_node_id, right_node_id])
    }

    pub fn new_subtract_node(
        &mut self,
        left_node_id: NodeId,
        right_node_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_subtract(&self.get_nodes(&[left_node_id, right_node_id])?)?;
        self.add_node_to_list(handle, name, "subtract", &[left_node_id, right_node_id])
    }

    pub fn new_flatten_node(
        &mut self,
        parent_id: NodeId,
        keep_first_dim: bool,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_flatten(&self.get_nodes(&[parent_id])?, keep_first_dim)?;
        self.add_node_to_list(handle, name, "flatten", &[parent_id])
    }

    pub fn new_reshape_node(
        &mut self,
        parent_id: NodeId,
        target_shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_reshape(&self.get_nodes(&[parent_id])?, target_shape)?;
        self.add_node_to_list(handle, name, "reshape", &[parent_id])
    }

    pub fn new_abs_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_abs(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "abs", &[parent_id])
    }

    pub fn new_sign_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_sign(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "sign", &[parent_id])
    }

    pub fn new_step_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_step(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "step", &[parent_id])
    }

    pub fn new_tanh_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_tanh(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "tanh", &[parent_id])
    }

    /// 创建 Sum 节点（归约求和）
    ///
    /// # 参数
    /// - `parent_id`: 输入节点 ID
    /// - `axis`: 求和轴，None 表示全局求和，Some(i) 表示沿轴 i 求和
    /// - `name`: 可选的节点名称
    ///
    /// # 用途
    /// - 全局求和：将所有元素求和为标量 [1, 1]
    /// - 按轴求和：SAC Actor Loss 中对动作维度求和 `Σ_a π(a|s) * (...)`
    pub fn new_sum_node(
        &mut self,
        parent_id: NodeId,
        axis: Option<usize>,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_sum(&self.get_nodes(&[parent_id])?, axis)?;
        self.add_node_to_list(handle, name, "sum", &[parent_id])
    }

    /// 创建 Mean 节点（归约求均值）
    ///
    /// # 参数
    /// - `parent_id`: 输入节点 ID
    /// - `axis`: 求均值轴，None 表示全局均值，Some(i) 表示沿轴 i 求均值
    /// - `name`: 可选的节点名称
    ///
    /// # 用途
    /// - 全局均值：将所有元素求均值为标量 [1, 1]
    /// - 按轴均值：如 batch 维度求均值以计算 loss
    pub fn new_mean_node(
        &mut self,
        parent_id: NodeId,
        axis: Option<usize>,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_mean(&self.get_nodes(&[parent_id])?, axis)?;
        self.add_node_to_list(handle, name, "mean", &[parent_id])
    }

    pub fn new_select_node(
        &mut self,
        parent_id: NodeId,
        axis: usize,
        index: usize,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_select(&self.get_nodes(&[parent_id])?, axis, index)?;
        self.add_node_to_list(handle, name, "select", &[parent_id])
    }

    /// 创建 Gather 节点（按索引张量从指定维度收集元素）
    ///
    /// # 参数
    /// - `input_id`: 输入数据节点 ID
    /// - `index_id`: 索引节点 ID
    /// - `dim`: gather 的维度
    /// - `name`: 可选的节点名称
    pub fn new_gather_node(
        &mut self,
        input_id: NodeId,
        index_id: NodeId,
        dim: usize,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_gather(&self.get_nodes(&[input_id, index_id])?, dim)?;
        self.add_node_to_list(handle, name, "gather", &[input_id, index_id])
    }

    /// 创建 Maximum 节点（逐元素取最大值）
    ///
    /// # 参数
    /// - `a_id`: 第一个输入节点 ID
    /// - `b_id`: 第二个输入节点 ID
    /// - `name`: 可选的节点名称
    ///
    /// # 用途
    /// PPO/TD3 等需要可微分 max 操作的场景
    pub fn new_maximum_node(
        &mut self,
        a_id: NodeId,
        b_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_maximum(&self.get_nodes(&[a_id, b_id])?)?;
        self.add_node_to_list(handle, name, "maximum", &[a_id, b_id])
    }

    /// 创建 Minimum 节点（逐元素取最小值）
    ///
    /// # 参数
    /// - `a_id`: 第一个输入节点 ID
    /// - `b_id`: 第二个输入节点 ID
    /// - `name`: 可选的节点名称
    ///
    /// # 用途
    /// PPO clipping、TD3 双 Q 网络等需要可微分 min 操作的场景
    pub fn new_minimum_node(
        &mut self,
        a_id: NodeId,
        b_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_minimum(&self.get_nodes(&[a_id, b_id])?)?;
        self.add_node_to_list(handle, name, "minimum", &[a_id, b_id])
    }

    /// 创建 Amax 节点（沿指定轴取最大值，只返回值不返回索引）
    ///
    /// # 参数
    /// - `input_id`: 输入节点 ID
    /// - `axis`: reduction 的轴
    /// - `name`: 可选的节点名称
    ///
    /// # 用途
    /// DQN 选最优动作 Q 值、特征池化等场景
    pub fn new_amax_node(
        &mut self,
        input_id: NodeId,
        axis: usize,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_amax(&self.get_nodes(&[input_id])?, axis)?;
        self.add_node_to_list(handle, name, "amax", &[input_id])
    }

    /// 创建 Amin 节点（沿指定轴取最小值，只返回值不返回索引）
    ///
    /// # 参数
    /// - `input_id`: 输入节点 ID
    /// - `axis`: reduction 的轴
    /// - `name`: 可选的节点名称
    ///
    /// # 用途
    /// Double DQN 选保守 Q 值、特征池化等场景
    pub fn new_amin_node(
        &mut self,
        input_id: NodeId,
        axis: usize,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_amin(&self.get_nodes(&[input_id])?, axis)?;
        self.add_node_to_list(handle, name, "amin", &[input_id])
    }

    pub fn new_sigmoid_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_sigmoid(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "sigmoid", &[parent_id])
    }

    /// 创建 Ln 节点（自然对数）
    ///
    /// # 参数
    /// - `parent_id`: 输入节点 ID
    /// - `name`: 可选的节点名称
    ///
    /// # 用途
    /// SAC 等需要计算 log π(a|s) 的强化学习算法
    pub fn new_ln_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_ln(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "ln", &[parent_id])
    }

    /// 创建 LogSoftmax 节点
    ///
    /// 沿最后一维计算数值稳定的 log(softmax(x))。
    /// 比 softmax().ln() 更稳定，避免小概率值导致的精度问题。
    ///
    /// # 参数
    /// - `parent_id`: 输入节点 ID，需要 2D 张量 [batch, num_classes]
    /// - `name`: 可选的节点名称
    ///
    /// # 用途
    /// - SAC Actor Loss 计算 log π(a|s)
    /// - 交叉熵损失等分类任务
    pub fn new_log_softmax_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_log_softmax(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "log_softmax", &[parent_id])
    }

    pub fn new_identity_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
        detached: bool,
    ) -> Result<NodeId, GraphError> {
        let mut handle = NodeHandle::new_identity(&self.get_nodes(&[parent_id])?)?;
        if detached {
            handle.set_detached(true);
        }
        self.add_node_to_list(handle, name, "identity", &[parent_id])
    }

    pub fn new_softmax_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_softmax(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "softmax", &[parent_id])
    }

    pub fn new_softplus_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_softplus(&self.get_nodes(&[parent_id])?)?;
        self.add_node_to_list(handle, name, "softplus", &[parent_id])
    }

    pub fn new_leaky_relu_node(
        &mut self,
        parent_id: NodeId,
        negative_slope: f32,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let handle = NodeHandle::new_leaky_relu(&self.get_nodes(&[parent_id])?, negative_slope)?;
        self.add_node_to_list(handle, name, "leaky_relu", &[parent_id])
    }

    pub fn new_relu_node(
        &mut self,
        parent_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        self.new_leaky_relu_node(parent_id, 0.0, name)
    }

    /// 创建 Dropout 节点
    ///
    /// # 参数
    /// - `parent_id`: 输入节点
    /// - `p`: 丢弃概率，范围 [0.0, 1.0)
    /// - `name`: 节点名称
    ///
    /// # 推荐值
    /// - 全连接层：0.5（经典值，可用 `DEFAULT_DROPOUT_P`）
    /// - 卷积层：0.1 ~ 0.3
    /// - 输入层后：0.1 ~ 0.2
    pub fn new_dropout_node(
        &mut self,
        parent_id: NodeId,
        p: f32,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        // 从 graph 的 rng 生成 seed（确保确定性）
        let seed = self.next_seed();
        let handle = NodeHandle::new_dropout(&self.get_nodes(&[parent_id])?, p, seed)?;
        self.add_node_to_list(handle, name, "dropout", &[parent_id])
    }

    pub fn new_softmax_cross_entropy_node(
        &mut self,
        logits_id: NodeId,
        labels_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[logits_id, labels_id])?;
        let handle = NodeHandle::new_softmax_cross_entropy(&parents)?;
        self.add_node_to_list(handle, name, "softmax_ce", &[logits_id, labels_id])
    }

    pub fn new_mse_loss_node(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_mse_loss(&parents)?;
        self.add_node_to_list(handle, name, "mse", &[input_id, target_id])
    }

    pub fn new_mse_loss_node_with_reduction(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        reduction: Reduction,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_mse_loss_with_reduction(&parents, reduction)?;
        self.add_node_to_list(handle, name, "mse", &[input_id, target_id])
    }

    pub fn new_mae_loss_node(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_mae_loss(&parents)?;
        self.add_node_to_list(handle, name, "mae", &[input_id, target_id])
    }

    pub fn new_mae_loss_node_with_reduction(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        reduction: Reduction,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_mae_loss_with_reduction(&parents, reduction)?;
        self.add_node_to_list(handle, name, "mae", &[input_id, target_id])
    }

    /// 创建 BCE（Binary Cross Entropy）损失节点（默认 Mean reduction）
    ///
    /// 采用 `BCEWithLogitsLoss` 形式，内置 Sigmoid 激活，数值稳定。
    /// 适用于二分类和多标签分类任务。
    ///
    /// # 参数
    /// - `logits_id`: 未激活的原始输出节点 ID
    /// - `target_id`: 二值标签节点 ID（0 或 1）
    /// - `name`: 节点名称（可选）
    pub fn new_bce_loss_node(
        &mut self,
        logits_id: NodeId,
        target_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[logits_id, target_id])?;
        let handle = NodeHandle::new_bce_loss(&parents)?;
        self.add_node_to_list(handle, name, "bce", &[logits_id, target_id])
    }

    /// 创建 BCE（Binary Cross Entropy）损失节点（指定 reduction 模式）
    pub fn new_bce_loss_node_with_reduction(
        &mut self,
        logits_id: NodeId,
        target_id: NodeId,
        reduction: Reduction,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[logits_id, target_id])?;
        let handle = NodeHandle::new_bce_loss_with_reduction(&parents, reduction)?;
        self.add_node_to_list(handle, name, "bce", &[logits_id, target_id])
    }

    /// 创建 Huber Loss 损失节点（默认 Mean reduction, δ=1.0）
    ///
    /// Huber Loss 结合 MSE（小误差）和 MAE（大误差）的优点：
    /// - |error| ≤ δ 时行为像 MSE（对小误差敏感）
    /// - |error| > δ 时行为像 MAE（对大误差鲁棒，梯度被"裁剪"到 ±δ）
    ///
    /// # 典型应用
    /// - **强化学习**：DQN 等算法的 Q 值训练（δ=1.0 是标准配置）
    /// - **带离群值的回归**：数据中存在异常值时
    ///
    /// # 参数
    /// - `input_id`: 预测值节点 ID
    /// - `target_id`: 目标值节点 ID
    /// - `name`: 节点名称（可选）
    pub fn new_huber_loss_node(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_huber_loss(&parents)?;
        self.add_node_to_list(handle, name, "huber", &[input_id, target_id])
    }

    /// 创建 Huber Loss 损失节点（指定 δ 参数）
    pub fn new_huber_loss_node_with_delta(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        delta: f32,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_huber_loss_with_delta(&parents, delta)?;
        self.add_node_to_list(handle, name, "huber", &[input_id, target_id])
    }

    /// 创建 Huber Loss 损失节点（完全自定义参数）
    pub fn new_huber_loss_node_with_params(
        &mut self,
        input_id: NodeId,
        target_id: NodeId,
        reduction: Reduction,
        delta: f32,
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let parents = self.get_nodes(&[input_id, target_id])?;
        let handle = NodeHandle::new_huber_loss_with_params(&parents, reduction, delta)?;
        self.add_node_to_list(handle, name, "huber", &[input_id, target_id])
    }
}
