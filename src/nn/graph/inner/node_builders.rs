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
use crate::nn::nodes::{NodeInner, NodeType};
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
        // 生成 ID
        let node_id = self.generate_valid_node_id();

        // 动态图节点命名：
        // - 用户指定名字：直接使用
        // - 自动命名：{type}_{批次内计数}，如 input_1, matmul_1
        //   计数器在每次 forward 完成后重置，确保同逻辑位置节点名字稳定
        let node_name = match name {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => {
                // 检查是否需要重置计数器（新批次开始）
                if self.last_forward_pass_id != self.counts_reset_pass_id {
                    self.node_type_counts.clear();
                    self.counts_reset_pass_id = self.last_forward_pass_id;
                }
                // 递增该类型的计数
                let count = self
                    .node_type_counts
                    .entry(node_type_str.to_string())
                    .or_insert(0);
                *count += 1;
                format!("{}_{}", node_type_str, *count)
            }
        };

        // 创建 NodeInner
        let node_inner = Rc::new(NodeInner::new(node_id, Some(node_name), raw_node, parents));

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

        // 2. 创建 Add 节点
        let add = Add::new(&parent_shapes_ref, &parent_dynamic_shapes)?;
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

        // 2. 使用 new 创建 Subtract 节点
        let subtract = Subtract::new(&parent_shapes_ref, &parent_dynamic_shapes, parent_ids)?;
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

        // 2. 使用 new 创建 Multiply 节点
        let multiply = Multiply::new(&parent_shapes_ref, &parent_dynamic_shapes, parent_ids)?;
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

        // 2. 使用 new 创建 Divide 节点
        let divide = Divide::new(&parent_shapes_ref, &parent_dynamic_shapes, parent_ids)?;
        let raw_node: NodeType = divide.into();

        // 3. 创建 NodeInner 并注册
        self.create_node_inner(raw_node, name, "divide", parents)
    }

    /// 创建 MatMul 节点（方案 C 新 API）
    ///
    /// 矩阵乘法 (left @ right)，要求 left 的列数等于 right 的行数。
    /// 返回 `Rc<NodeInner>`，父节点引用由 `parents` 参数传入。
    pub fn create_mat_mul_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::MatMul;

        // 1. 从父节点提取形状信息
        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        // 2. 使用 new 创建 MatMul 节点
        let mat_mul = MatMul::new(&parent_shapes_ref, &parent_dynamic_shapes, parent_ids)?;
        let raw_node: NodeType = mat_mul.into();

        // 3. 创建 NodeInner 并注册
        self.create_node_inner(raw_node, name, "matmul", parents)
    }

    /// 创建 Conv2d 节点（方案 C 新 API）
    ///
    /// 2D 卷积操作，输入必须是 4D [batch, C_in, H, W]。
    /// 返回 `Rc<NodeInner>`，父节点引用由 `parents` 参数传入。
    ///
    /// # 参数
    /// - `parents`: [输入节点, 卷积核节点]
    /// - `stride`: 步长 (sH, sW)
    /// - `padding`: 填充 (pH, pW)
    pub fn create_conv2d_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        stride: (usize, usize),
        padding: (usize, usize),
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Conv2d;

        // 1. 从父节点提取形状信息
        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        // 2. 使用 new 创建 Conv2d 节点
        let conv2d = Conv2d::new(
            &parent_shapes_ref,
            &parent_dynamic_shapes,
            parent_ids,
            stride,
            padding,
        )?;
        let raw_node: NodeType = conv2d.into();

        // 3. 创建 NodeInner 并注册
        self.create_node_inner(raw_node, name, "conv2d", parents)
    }

    /// 创建 MaxPool2d 节点（方案 C 新 API）
    ///
    /// 2D 最大池化操作，输入必须是 4D [batch, C, H, W]。
    /// 返回 `Rc<NodeInner>`，父节点引用由 `parents` 参数传入。
    ///
    /// # 参数
    /// - `parent`: 输入节点
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长，None 则默认等于 kernel_size
    pub fn create_max_pool2d_node(
        &mut self,
        parent: Rc<NodeInner>,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::MaxPool2d;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let max_pool = MaxPool2d::new(&parent_shape, &parent_dynamic_shape, kernel_size, stride)?;
        let raw_node: NodeType = max_pool.into();

        self.create_node_inner(raw_node, name, "maxpool2d", vec![parent])
    }

    /// 创建 AvgPool2d 节点（方案 C 新 API）
    ///
    /// 2D 平均池化操作，输入必须是 4D [batch, C, H, W]。
    /// 返回 `Rc<NodeInner>`，父节点引用由 `parents` 参数传入。
    ///
    /// # 参数
    /// - `parent`: 输入节点
    /// - `kernel_size`: 池化窗口大小 (kH, kW)
    /// - `stride`: 步长，None 则默认等于 kernel_size
    pub fn create_avg_pool2d_node(
        &mut self,
        parent: Rc<NodeInner>,
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::AvgPool2d;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let avg_pool = AvgPool2d::new(&parent_shape, &parent_dynamic_shape, kernel_size, stride)?;
        let raw_node: NodeType = avg_pool.into();

        self.create_node_inner(raw_node, name, "avgpool2d", vec![parent])
    }

    /// 创建 Flatten 节点（方案 C 新 API）
    ///
    /// 将输入展平为 2D 张量。
    /// - `keep_first_dim = true`: [d0, d1, d2, ...] → [d0, d1*d2*...]
    /// - `keep_first_dim = false`: [d0, d1, ...] → [1, d0*d1*...]
    pub fn create_flatten_node(
        &mut self,
        parent: Rc<NodeInner>,
        keep_first_dim: bool,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Flatten;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let flatten = Flatten::new(&parent_shape, &parent_dynamic_shape, keep_first_dim)?;
        let raw_node: NodeType = flatten.into();

        self.create_node_inner(raw_node, name, "flatten", vec![parent])
    }

    /// 创建 Reshape 节点（方案 C 新 API）
    ///
    /// 改变张量形状而不改变数据，目标形状元素总数必须与输入相同。
    pub fn create_reshape_node(
        &mut self,
        parent: Rc<NodeInner>,
        target_shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Reshape;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let reshape = Reshape::new(&parent_shape, &parent_dynamic_shape, target_shape)?;
        let raw_node: NodeType = reshape.into();

        self.create_node_inner(raw_node, name, "reshape", vec![parent])
    }

    /// 创建 Stack 节点（方案 C 新 API）
    ///
    /// 将多个张量沿指定轴堆叠/拼接。
    /// - `new_dim=true`: 在指定位置插入新维度后堆叠（类似 torch.stack）
    /// - `new_dim=false`: 沿现有维度拼接（类似 torch.cat）
    pub fn create_stack_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        axis: usize,
        new_dim: bool,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Stack;

        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        let stack = Stack::new(
            &parent_shapes_ref,
            &parent_dynamic_shapes,
            parent_ids,
            axis,
            new_dim,
        )?;
        let raw_node: NodeType = stack.into();

        self.create_node_inner(raw_node, name, "stack", parents)
    }

    /// 创建 Select 节点（方案 C 新 API）
    ///
    /// 从张量中选择指定轴和索引的切片，输出维度减少 1。
    pub fn create_select_node(
        &mut self,
        parent: Rc<NodeInner>,
        axis: usize,
        index: usize,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Select;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let select = Select::new(&parent_shape, &parent_dynamic_shape, axis, index)?;
        let raw_node: NodeType = select.into();

        self.create_node_inner(raw_node, name, "select", vec![parent])
    }

    /// 创建 Gather 节点（方案 C 新 API）
    ///
    /// 按索引张量从指定维度收集元素。
    /// 用于 SAC/DQN 等强化学习算法：按动作索引选择 Q 值。
    ///
    /// # 参数
    /// - `input`: 输入数据节点（如 Q 值）
    /// - `index`: 索引节点（如动作索引）
    /// - `dim`: gather 的维度
    pub fn create_gather_node(
        &mut self,
        input: Rc<NodeInner>,
        index: Rc<NodeInner>,
        dim: usize,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Gather;

        let input_shape = input.shape();
        let index_shape = index.shape();
        let input_dynamic_shape = input.dynamic_shape();
        let index_dynamic_shape = index.dynamic_shape();

        let gather = Gather::new(
            &input_shape,
            &index_shape,
            &input_dynamic_shape,
            &index_dynamic_shape,
            dim,
        )?;
        let raw_node: NodeType = gather.into();

        self.create_node_inner(raw_node, name, "gather", vec![input, index])
    }

    /// 创建 Sigmoid 节点（方案 C 新 API）
    ///
    /// Sigmoid 激活函数：sigmoid(x) = 1 / (1 + e^(-x))
    pub fn create_sigmoid_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Sigmoid;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let sigmoid = Sigmoid::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = sigmoid.into();

        self.create_node_inner(raw_node, name, "sigmoid", vec![parent])
    }

    /// 创建 Tanh 节点（方案 C 新 API）
    ///
    /// Tanh 激活函数：tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    pub fn create_tanh_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Tanh;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let tanh = Tanh::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = tanh.into();

        self.create_node_inner(raw_node, name, "tanh", vec![parent])
    }

    /// 创建 LeakyReLU 节点（方案 C 新 API）
    ///
    /// LeakyReLU: f(x) = x if x > 0, else negative_slope * x
    pub fn create_leaky_relu_node(
        &mut self,
        parent: Rc<NodeInner>,
        negative_slope: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::LeakyReLU;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let leaky_relu = LeakyReLU::new(&parent_shape, &parent_dynamic_shape, negative_slope)?;
        let raw_node: NodeType = leaky_relu.into();

        self.create_node_inner(raw_node, name, "leaky_relu", vec![parent])
    }

    /// 创建 ReLU 节点（方案 C 新 API）
    ///
    /// ReLU: f(x) = max(0, x)，等价于 LeakyReLU(negative_slope=0)
    pub fn create_relu_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        self.create_leaky_relu_node(parent, 0.0, name)
    }

    /// 创建 Softmax 节点（方案 C 新 API）
    ///
    /// Softmax: softmax(x)_i = exp(x_i) / Σ exp(x_j)，沿最后一维归一化
    pub fn create_softmax_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Softmax;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let softmax = Softmax::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = softmax.into();

        self.create_node_inner(raw_node, name, "softmax", vec![parent])
    }

    /// 创建 LogSoftmax 节点（方案 C 新 API）
    ///
    /// LogSoftmax: log(softmax(x))，数值稳定版本
    pub fn create_log_softmax_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::LogSoftmax;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let log_softmax = LogSoftmax::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = log_softmax.into();

        self.create_node_inner(raw_node, name, "log_softmax", vec![parent])
    }

    /// 创建 SoftPlus 节点（方案 C 新 API）
    ///
    /// SoftPlus: f(x) = ln(1 + e^x)，ReLU 的平滑近似
    pub fn create_softplus_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::SoftPlus;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let softplus = SoftPlus::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = softplus.into();

        self.create_node_inner(raw_node, name, "softplus", vec![parent])
    }

    /// 创建 MSE 损失节点（方案 C 新 API）
    ///
    /// MSE: mean((input - target)^2)
    pub fn create_mse_node(
        &mut self,
        input: Rc<NodeInner>,
        target: Rc<NodeInner>,
        reduction: crate::nn::nodes::raw_node::Reduction,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::MSE;

        let input_shape = input.shape();
        let target_shape = target.shape();
        let input_dynamic_shape = input.dynamic_shape();
        let target_dynamic_shape = target.dynamic_shape();

        let mse = MSE::new(
            &input_shape,
            &target_shape,
            &input_dynamic_shape,
            &target_dynamic_shape,
            vec![input.id(), target.id()],
            reduction,
        )?;
        let raw_node: NodeType = mse.into();

        self.create_node_inner(raw_node, name, "mse", vec![input, target])
    }

    /// 创建 MSE 损失节点（Mean reduction，方案 C 新 API）
    pub fn create_mse_mean_node(
        &mut self,
        input: Rc<NodeInner>,
        target: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        self.create_mse_node(
            input,
            target,
            crate::nn::nodes::raw_node::Reduction::Mean,
            name,
        )
    }

    /// 创建 MAE 损失节点（方案 C 新 API）
    ///
    /// MAE: mean(|input - target|)
    pub fn create_mae_node(
        &mut self,
        input: Rc<NodeInner>,
        target: Rc<NodeInner>,
        reduction: crate::nn::nodes::raw_node::Reduction,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::MAE;

        let input_shape = input.shape();
        let target_shape = target.shape();
        let input_dynamic_shape = input.dynamic_shape();
        let target_dynamic_shape = target.dynamic_shape();

        let mae = MAE::new(
            &input_shape,
            &target_shape,
            &input_dynamic_shape,
            &target_dynamic_shape,
            vec![input.id(), target.id()],
            reduction,
        )?;
        let raw_node: NodeType = mae.into();

        self.create_node_inner(raw_node, name, "mae", vec![input, target])
    }

    /// 创建 MAE 损失节点（Mean reduction，方案 C 新 API）
    pub fn create_mae_mean_node(
        &mut self,
        input: Rc<NodeInner>,
        target: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        self.create_mae_node(
            input,
            target,
            crate::nn::nodes::raw_node::Reduction::Mean,
            name,
        )
    }

    /// 创建 BCE 损失节点（方案 C 新 API）
    ///
    /// BCE: Binary Cross Entropy with Logits
    pub fn create_bce_node(
        &mut self,
        logits: Rc<NodeInner>,
        target: Rc<NodeInner>,
        reduction: crate::nn::nodes::raw_node::Reduction,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::BCE;

        let logits_shape = logits.shape();
        let target_shape = target.shape();
        let logits_dynamic_shape = logits.dynamic_shape();
        let target_dynamic_shape = target.dynamic_shape();

        let bce = BCE::new(
            &logits_shape,
            &target_shape,
            &logits_dynamic_shape,
            &target_dynamic_shape,
            vec![logits.id(), target.id()],
            reduction,
        )?;
        let raw_node: NodeType = bce.into();

        self.create_node_inner(raw_node, name, "bce", vec![logits, target])
    }

    /// 创建 BCE 损失节点（Mean reduction，方案 C 新 API）
    pub fn create_bce_mean_node(
        &mut self,
        logits: Rc<NodeInner>,
        target: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        self.create_bce_node(
            logits,
            target,
            crate::nn::nodes::raw_node::Reduction::Mean,
            name,
        )
    }

    /// 创建 Huber 损失节点（方案 C 新 API）
    ///
    /// Huber Loss: 结合 MSE 和 MAE 的优点
    pub fn create_huber_node(
        &mut self,
        input: Rc<NodeInner>,
        target: Rc<NodeInner>,
        reduction: crate::nn::nodes::raw_node::Reduction,
        delta: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Huber;

        let input_shape = input.shape();
        let target_shape = target.shape();
        let input_dynamic_shape = input.dynamic_shape();
        let target_dynamic_shape = target.dynamic_shape();

        let huber = Huber::new(
            &input_shape,
            &target_shape,
            &input_dynamic_shape,
            &target_dynamic_shape,
            vec![input.id(), target.id()],
            reduction,
            delta,
        )?;
        let raw_node: NodeType = huber.into();

        self.create_node_inner(raw_node, name, "huber", vec![input, target])
    }

    /// 创建 Huber 损失节点（默认参数：Mean reduction, δ=1.0）
    pub fn create_huber_default_node(
        &mut self,
        input: Rc<NodeInner>,
        target: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        self.create_huber_node(
            input,
            target,
            crate::nn::nodes::raw_node::Reduction::Mean,
            crate::nn::nodes::raw_node::DEFAULT_HUBER_DELTA,
            name,
        )
    }

    /// 创建 SoftmaxCrossEntropy 损失节点（方案 C 新 API）
    ///
    /// 融合 Softmax + CrossEntropy，数值稳定
    pub fn create_softmax_cross_entropy_node(
        &mut self,
        logits: Rc<NodeInner>,
        labels: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::SoftmaxCrossEntropy;

        let logits_shape = logits.shape();
        let labels_shape = labels.shape();
        let logits_dynamic_shape = logits.dynamic_shape();
        let labels_dynamic_shape = labels.dynamic_shape();

        let sce = SoftmaxCrossEntropy::new(
            &logits_shape,
            &labels_shape,
            &logits_dynamic_shape,
            &labels_dynamic_shape,
            vec![logits.id(), labels.id()],
        )?;
        let raw_node: NodeType = sce.into();

        self.create_node_inner(raw_node, name, "softmax_ce", vec![logits, labels])
    }

    // ==================== 归约节点（方案 C 新 API）====================

    /// 创建 Sum 归约节点（方案 C 新 API）
    ///
    /// - `axis = None`：全局求和，输出 [1, 1]
    /// - `axis = Some(i)`：沿轴 i 求和（keepdims=true）
    pub fn create_sum_node(
        &mut self,
        input: Rc<NodeInner>,
        axis: Option<usize>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Sum;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let sum = Sum::new(&input_shape, &input_dynamic_shape, axis)?;
        let raw_node: NodeType = sum.into();

        self.create_node_inner(raw_node, name, "sum", vec![input])
    }

    /// 创建 Mean 归约节点（方案 C 新 API）
    ///
    /// - `axis = None`：全局均值，输出 [1, 1]
    /// - `axis = Some(i)`：沿轴 i 均值（keepdims=true）
    pub fn create_mean_node(
        &mut self,
        input: Rc<NodeInner>,
        axis: Option<usize>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Mean;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let mean = Mean::new(&input_shape, &input_dynamic_shape, axis)?;
        let raw_node: NodeType = mean.into();

        self.create_node_inner(raw_node, name, "mean", vec![input])
    }

    /// 创建 Amax 归约节点（方案 C 新 API）
    ///
    /// 沿指定轴取最大值（移除该轴）
    pub fn create_amax_node(
        &mut self,
        input: Rc<NodeInner>,
        axis: usize,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Amax;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let amax = Amax::new(&input_shape, &input_dynamic_shape, axis)?;
        let raw_node: NodeType = amax.into();

        self.create_node_inner(raw_node, name, "amax", vec![input])
    }

    /// 创建 Amin 归约节点（方案 C 新 API）
    ///
    /// 沿指定轴取最小值（移除该轴）
    pub fn create_amin_node(
        &mut self,
        input: Rc<NodeInner>,
        axis: usize,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Amin;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let amin = Amin::new(&input_shape, &input_dynamic_shape, axis)?;
        let raw_node: NodeType = amin.into();

        self.create_node_inner(raw_node, name, "amin", vec![input])
    }

    /// 创建 Maximum 节点（方案 C 新 API）
    ///
    /// 逐元素取两个张量的最大值
    pub fn create_maximum_node(
        &mut self,
        a: Rc<NodeInner>,
        b: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Maximum;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let a_dynamic_shape = a.dynamic_shape();
        let b_dynamic_shape = b.dynamic_shape();

        let maximum = Maximum::new(&a_shape, &b_shape, &a_dynamic_shape, &b_dynamic_shape)?;
        let raw_node: NodeType = maximum.into();

        self.create_node_inner(raw_node, name, "maximum", vec![a, b])
    }

    /// 创建 Minimum 节点（方案 C 新 API）
    ///
    /// 逐元素取两个张量的最小值
    pub fn create_minimum_node(
        &mut self,
        a: Rc<NodeInner>,
        b: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Minimum;

        let a_shape = a.shape();
        let b_shape = b.shape();
        let a_dynamic_shape = a.dynamic_shape();
        let b_dynamic_shape = b.dynamic_shape();

        let minimum = Minimum::new(&a_shape, &b_shape, &a_dynamic_shape, &b_dynamic_shape)?;
        let raw_node: NodeType = minimum.into();

        self.create_node_inner(raw_node, name, "minimum", vec![a, b])
    }

    // ==================== 其他节点（方案 C 新 API）====================

    /// 创建 Identity 节点（方案 C 新 API）
    ///
    /// 恒等映射，用于梯度截断边界
    ///
    /// # 参数
    /// - `input`: 输入节点
    /// - `name`: 节点名称
    /// - `detached`: 是否设置为 detached（梯度截断）
    ///
    /// # detached 说明
    /// - `detached = false`: 正常 Identity，梯度正常传播
    /// - `detached = true`: 梯度截断，用于 `Var::detach()`
    pub fn create_identity_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
        detached: bool,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Identity;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let identity = Identity::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = identity.into();

        let node = self.create_node_inner(raw_node, name, "identity", vec![input])?;

        // 设置 detached 状态
        if detached {
            node.set_detached(true);
        }

        Ok(node)
    }

    /// 创建 Dropout 节点（方案 C 新 API）
    ///
    /// 训练时随机丢弃部分神经元
    pub fn create_dropout_node(
        &mut self,
        input: Rc<NodeInner>,
        p: f32,
        seed: u64,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Dropout;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let dropout = Dropout::new(&input_shape, &input_dynamic_shape, p, seed)?;
        let raw_node: NodeType = dropout.into();

        self.create_node_inner(raw_node, name, "dropout", vec![input])
    }

    /// 创建 ZerosLike 节点（方案 C 新 API）
    ///
    /// 以 `reference` 为父节点，前向传播时读取其 batch_size 生成零张量。
    /// 反向传播时 `calc_grad_to_parent` 返回 `InvalidOperation("不支持")`，
    /// 被 `propagate_grad_to_parents` 的 `continue` 逻辑安全跳过。
    pub fn create_zeros_like_node(
        &mut self,
        reference: Rc<NodeInner>,
        feature_shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::ZerosLike;

        let zeros_like = ZerosLike::new(feature_shape);
        let raw_node: NodeType = zeros_like.into();

        self.create_node_inner(raw_node, name, "zeros_like", vec![reference])
    }

    /// 创建 Abs 节点（方案 C 新 API）
    ///
    /// 逐元素绝对值
    pub fn create_abs_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Abs;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let abs = Abs::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = abs.into();

        self.create_node_inner(raw_node, name, "abs", vec![input])
    }

    /// 创建 Sign 节点（方案 C 新 API）
    ///
    /// 逐元素符号函数
    pub fn create_sign_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Sign;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let sign = Sign::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = sign.into();

        self.create_node_inner(raw_node, name, "sign", vec![input])
    }

    /// 创建 Step 节点（方案 C 新 API）
    ///
    /// 逐元素阶跃函数
    pub fn create_step_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Step;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let step = Step::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = step.into();

        self.create_node_inner(raw_node, name, "step", vec![input])
    }

    /// 创建 Ln 节点（方案 C 新 API）
    ///
    /// 逐元素自然对数
    pub fn create_ln_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Ln;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let ln = Ln::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = ln.into();

        self.create_node_inner(raw_node, name, "ln", vec![input])
    }
}
