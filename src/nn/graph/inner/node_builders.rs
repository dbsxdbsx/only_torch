/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 节点构建方法（new_*_node）
 *
 * - create_node_inner(): 通用节点创建，返回 Rc<NodeInner>
 * - create_basic_input_node() 等: 各节点类型的新创建方法
 */

use super::super::error::GraphError;
use super::GraphInner;
use crate::nn::NodeId;
use crate::nn::nodes::{NodeInner, NodeType};
use std::rc::Rc;

impl GraphInner {
    // ==================== 节点创建 API ====================

    /// 创建 NodeInner
    ///
    /// # 参数
    /// - `raw_node`: 节点类型（NodeType）
    /// - `name`: 可选的节点名称
    /// - `node_type_str`: 节点类型字符串（用于生成默认名称）
    /// - `parents`: 父节点的强引用列表
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
        use crate::nn::nodes::raw_node::TraitNode;

        // ===== CSE：计算去重 key（raw_node move 前）=====
        // 条件：自动命名（name 为空）且节点类型参与去重（fingerprint 非 None）
        let dedup_key = if name.is_none() || name == Some("") {
            raw_node.dedup_fingerprint().map(|fp| {
                // 缓存随 forward_pass_id 变化清空（同 node_type_counts 机制）
                if self.last_forward_pass_id != self.cse_cache_reset_pass_id {
                    self.cse_cache.clear();
                    self.cse_cache_reset_pass_id = self.last_forward_pass_id;
                }
                let parent_ids: Vec<_> = parents.iter().map(|p| p.id()).collect();
                let group_ctx = self.node_group_context.clone();
                (node_type_str.to_string(), parent_ids, fp, group_ctx)
            })
        } else {
            None
        };

        // ===== CSE：查缓存 =====
        if let Some(ref key) = dedup_key {
            if let Some(weak) = self.cse_cache.get(key) {
                if let Some(existing) = weak.upgrade() {
                    return Ok(existing); // 命中：复用已有节点，raw_node 被 drop
                }
            }
        }

        // ===== 原有逻辑：生成 ID、命名、创建节点 =====
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

        // 同步 ID 和名称到 raw_node（用于错误消息中的 display_node()）
        let mut raw_node = raw_node;
        raw_node.set_id(node_id);
        raw_node.set_name(&node_name);

        // 节点分组自动标记：如果当前有活跃的分组上下文，
        // 给计算节点打标签。
        // - Input 节点始终排除（外部数据）
        // - Parameter 节点：当 node_group_include_params=true 时标记（Layer/Recurrent），
        //   否则排除（Distribution）
        let group_tag = if self.node_group_context.is_some() {
            let is_input = matches!(raw_node, NodeType::Input(_));
            let is_param = matches!(raw_node, NodeType::Parameter(_));
            if is_input || (is_param && !self.node_group_include_params) {
                None
            } else {
                self.node_group_context.clone()
            }
        } else {
            None
        };

        // 创建 NodeInner
        let node = NodeInner::new(node_id, Some(node_name), raw_node, parents);
        if group_tag.is_some() {
            node.set_node_group_tag(group_tag);
        }

        let node_inner = Rc::new(node);

        // ===== CSE：写入缓存 =====
        if let Some(key) = dedup_key {
            self.cse_cache
                .insert(key, Rc::downgrade(&node_inner));
        }

        Ok(node_inner)
    }

    /// 创建 Data 输入节点
    ///
    /// 用户数据入口，可视化中显示为蓝色椭圆。
    /// 返回 `Rc<NodeInner>`，叶子节点（无父节点）。
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

    /// 创建 Target 输入节点
    ///
    /// Loss 的目标值（真实标签），可视化中显示为橙色椭圆。
    /// 底层与 Data 共用 BasicInput，区别仅在于 InputVariant 变体。
    pub fn create_target_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::InputVariant;

        let input_variant = InputVariant::new_target(shape)?;
        let raw_node: NodeType = input_variant.into();

        self.create_node_inner(raw_node, name, "target", vec![])
    }

    /// 创建 Parameter 节点    ///
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

    /// 创建带固定种子的 Parameter 节点    ///
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

    /// 创建 State 节点    ///
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

    /// 创建 Add 节点    ///
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

    /// 创建 Subtract 节点    ///
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

    /// 创建 Multiply 节点    ///
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

    /// 创建 Divide 节点    ///
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

    /// 创建 Negate 节点    ///
    /// 逐元素取反 (-x)
    pub fn create_negate_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Negate;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let negate = Negate::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = negate.into();

        self.create_node_inner(raw_node, name, "negate", vec![input])
    }

    /// 创建 MatMul 节点    ///
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

    /// 创建 Conv2d 节点    ///
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
        dilation: (usize, usize),
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Conv2d;

        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        let conv2d = Conv2d::new(
            &parent_shapes_ref,
            &parent_dynamic_shapes,
            parent_ids,
            stride,
            padding,
            dilation,
        )?;
        let raw_node: NodeType = conv2d.into();

        self.create_node_inner(raw_node, name, "conv2d", parents)
    }

    /// 创建 ConvTranspose2d（转置卷积）节点
    pub fn create_conv_transpose2d_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::ConvTranspose2d;

        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        let node = ConvTranspose2d::new(
            &parent_shapes_ref,
            &parent_dynamic_shapes,
            parent_ids,
            stride,
            padding,
            output_padding,
        )?;
        let raw_node: NodeType = node.into();

        self.create_node_inner(raw_node, name, "conv_transpose2d", parents)
    }

    /// 创建 MaxPool2d 节点
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

    /// 创建 AvgPool2d 节点    ///
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

    /// 创建 Upsample2d 节点（最近邻上采样）
    ///
    /// 2D 最近邻上采样，输入必须是 4D [batch, C, H, W]，
    /// 输出形状为 [batch, C, H*scale_h, W*scale_w]。
    ///
    /// 用途：YOLOv5 等目标检测网络的 PAN/FPN 颈部，
    /// 把深层小特征图上采样后跟浅层特征图拼接做多尺度融合。
    ///
    /// # 参数
    /// - `parent`: 输入节点
    /// - `scale_h`: H 方向放大倍数（必须 ≥ 1）
    /// - `scale_w`: W 方向放大倍数（必须 ≥ 1）
    pub fn create_upsample2d_node(
        &mut self,
        parent: Rc<NodeInner>,
        scale_h: usize,
        scale_w: usize,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Upsample2d;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let upsample = Upsample2d::new(&parent_shape, &parent_dynamic_shape, scale_h, scale_w)?;
        let raw_node: NodeType = upsample.into();

        self.create_node_inner(raw_node, name, "upsample2d", vec![parent])
    }

    /// 创建 Flatten 节点    ///
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

    /// 创建 Reshape 节点    ///
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

    /// 创建 Stack 节点
    ///
    /// 在指定位置插入新维度后堆叠（类似 `torch.stack`）。
    /// 所有父节点形状必须完全相同。
    pub fn create_stack_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        axis: usize,
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
        )?;
        let raw_node: NodeType = stack.into();

        self.create_node_inner(raw_node, name, "stack", parents)
    }

    /// 创建 Concat 节点
    ///
    /// 沿现有维度拼接（类似 `torch.cat` / `tf.concat`）。
    /// 除 `axis` 外其他维度必须相同。
    pub fn create_concat_node(
        &mut self,
        parents: Vec<Rc<NodeInner>>,
        axis: usize,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Concat;

        let parent_shapes: Vec<Vec<usize>> = parents.iter().map(|p| p.shape()).collect();
        let parent_shapes_ref: Vec<&[usize]> = parent_shapes.iter().map(|s| s.as_slice()).collect();
        let parent_dynamic_shapes: Vec<_> = parents.iter().map(|p| p.dynamic_shape()).collect();
        let parent_ids: Vec<NodeId> = parents.iter().map(|p| p.id()).collect();

        let concat = Concat::new(
            &parent_shapes_ref,
            &parent_dynamic_shapes,
            parent_ids,
            axis,
        )?;
        let raw_node: NodeType = concat.into();

        self.create_node_inner(raw_node, name, "concat", parents)
    }

    /// 创建 Select 节点    ///
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

    /// 创建 Gather 节点    ///
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

    /// 创建 Narrow 节点
    ///
    /// Narrow: 沿单轴取连续子范围（不降维）
    pub fn create_narrow_node(
        &mut self,
        parent: Rc<NodeInner>,
        axis: usize,
        start: usize,
        length: usize,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Narrow;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let narrow = Narrow::new(&parent_shape, &parent_dynamic_shape, axis, start, length)?;
        let raw_node: NodeType = narrow.into();

        self.create_node_inner(raw_node, name, "narrow", vec![parent])
    }

    /// 创建 Permute 节点
    ///
    /// 维度重排（转置的一般形式）：output = input.permute(dims)
    pub fn create_permute_node(
        &mut self,
        parent: Rc<NodeInner>,
        dims: &[usize],
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Permute;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let permute = Permute::new(&parent_shape, &parent_dynamic_shape, dims)?;
        let raw_node: NodeType = permute.into();

        self.create_node_inner(raw_node, name, "permute", vec![parent])
    }

    /// 创建 TopK 节点
    ///
    /// 沿指定轴选取前 k 大元素。
    /// forward 输出 values，内部保存 indices 用于 backward scatter。
    pub fn create_topk_node(
        &mut self,
        parent: Rc<NodeInner>,
        k: usize,
        axis: usize,
        sorted: bool,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::TopK;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let topk = TopK::new(&parent_shape, &parent_dynamic_shape, k, axis, sorted)?;
        let raw_node: NodeType = topk.into();

        self.create_node_inner(raw_node, name, "topk", vec![parent])
    }

    /// 创建 Sigmoid 节点    ///
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

    /// 创建 Tanh 节点    ///
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

    /// 创建 LeakyReLU 节点    ///
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

    /// 创建 ReLU 节点
    ///
    /// ReLU: f(x) = max(0, x)
    pub fn create_relu_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::ReLU;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let relu = ReLU::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = relu.into();

        self.create_node_inner(raw_node, name, "relu", vec![parent])
    }

    /// 创建 Softmax 节点    ///
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

    /// 创建 LogSoftmax 节点    ///
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

    /// 创建 GELU 节点
    ///
    /// GELU: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn create_gelu_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Gelu;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let gelu = Gelu::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = gelu.into();

        self.create_node_inner(raw_node, name, "gelu", vec![parent])
    }

    /// 创建 Swish/SiLU 节点
    ///
    /// Swish: swish(x) = x * sigmoid(x)
    pub fn create_swish_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Swish;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let swish = Swish::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = swish.into();

        self.create_node_inner(raw_node, name, "swish", vec![parent])
    }

    /// 创建 ELU 节点
    ///
    /// ELU: elu(x, alpha) = x if x > 0, else alpha * (exp(x) - 1)
    pub fn create_elu_node(
        &mut self,
        parent: Rc<NodeInner>,
        alpha: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Elu;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let elu = Elu::new(&parent_shape, &parent_dynamic_shape, alpha)?;
        let raw_node: NodeType = elu.into();

        self.create_node_inner(raw_node, name, "elu", vec![parent])
    }

    /// 创建 SELU 节点
    ///
    /// SELU: selu(x) = LAMBDA * elu(x, ALPHA)
    pub fn create_selu_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Selu;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let selu = Selu::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = selu.into();

        self.create_node_inner(raw_node, name, "selu", vec![parent])
    }

    /// 创建 Mish 节点
    ///
    /// Mish: mish(x) = x * tanh(softplus(x))
    pub fn create_mish_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Mish;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let mish = Mish::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = mish.into();

        self.create_node_inner(raw_node, name, "mish", vec![parent])
    }

    /// 创建 HardSwish 节点
    ///
    /// HardSwish: 分段函数，CPU 友好（无 exp）
    pub fn create_hard_swish_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::HardSwish;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let hard_swish = HardSwish::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = hard_swish.into();

        self.create_node_inner(raw_node, name, "hard_swish", vec![parent])
    }

    /// 创建 HardSigmoid 节点
    ///
    /// HardSigmoid: 分段函数，CPU 友好（无 exp）
    pub fn create_hard_sigmoid_node(
        &mut self,
        parent: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::HardSigmoid;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let hard_sigmoid = HardSigmoid::new(&parent_shape, &parent_dynamic_shape)?;
        let raw_node: NodeType = hard_sigmoid.into();

        self.create_node_inner(raw_node, name, "hard_sigmoid", vec![parent])
    }

    /// 创建 SoftPlus 节点    ///
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

    /// 创建 MSE 损失节点    ///
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

    /// 创建 MSE 损失节点（Mean reduction）
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

    /// 创建 MAE 损失节点    ///
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

    /// 创建 MAE 损失节点（Mean reduction）
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

    /// 创建 BCE 损失节点    ///
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

    /// 创建 BCE 损失节点（Mean reduction）
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

    /// 创建 Huber 损失节点    ///
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

    /// 创建 SoftmaxCrossEntropy 损失节点    ///
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

    // ==================== 归约节点 ====================

    /// 创建 Sum 归约节点    ///
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

    /// 创建 Mean 归约节点    ///
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

    /// 创建 Amax 归约节点    ///
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

    /// 创建 Amin 归约节点    ///
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

    /// 创建 Maximum 节点    ///
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

    /// 创建 Minimum 节点    ///
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

    // ==================== 其他节点 ====================

    /// 创建 Identity 节点（纯恒等映射）
    ///
    /// 前向传播透传值，反向传播透传梯度。
    /// 用于 NEAT 占位、skip connection 等场景。
    pub fn create_identity_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Identity;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let identity = Identity::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = identity.into();

        self.create_node_inner(raw_node, name, "identity", vec![input])
    }

    /// 创建 Detach 节点（梯度屏障）
    ///
    /// 前向传播透传值，反向传播**阻断**梯度。
    /// 通过 `Var::detach()` 调用。
    pub fn create_detach_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Detach;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let detach = Detach::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = detach.into();

        self.create_node_inner(raw_node, name, "detach", vec![input])
    }

    /// 创建 Dropout 节点    ///
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

    /// 创建 ZerosLike 节点    ///
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

    /// 创建 Abs 节点    ///
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

    /// 创建 Sign 节点    ///
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

    /// 创建 Step 节点    ///
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

    /// 创建 Ln 节点    ///
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

    /// 创建 Sqrt 节点
    ///
    /// 逐元素平方根: y = √x
    pub fn create_sqrt_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Sqrt;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let sqrt = Sqrt::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = sqrt.into();

        self.create_node_inner(raw_node, name, "sqrt", vec![input])
    }

    /// 创建 BatchNormOp 节点
    ///
    /// 批归一化核心计算（不含 gamma/beta）
    ///
    /// # 参数
    /// - `running_mean`/`running_var`: 共享的 running stats，由 BatchNorm 层持有，
    ///   跨 forward 调用持久化 EMA 统计量。
    pub fn create_batch_norm_op_node(
        &mut self,
        input: Rc<NodeInner>,
        eps: f32,
        momentum: f32,
        running_mean: std::rc::Rc<std::cell::RefCell<crate::tensor::Tensor>>,
        running_var: std::rc::Rc<std::cell::RefCell<crate::tensor::Tensor>>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::BatchNormOp;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let bn = BatchNormOp::new(
            &input_shape, &input_dynamic_shape, eps, momentum,
            running_mean, running_var,
        )?;
        let raw_node: NodeType = bn.into();

        self.create_node_inner(raw_node, name, "batch_norm", vec![input])
    }

    /// 创建 LayerNormOp 节点
    ///
    /// 层归一化核心计算（不含 gamma/beta）
    pub fn create_layer_norm_op_node(
        &mut self,
        input: Rc<NodeInner>,
        normalized_dims: usize,
        eps: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::LayerNormOp;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let ln = LayerNormOp::new(&input_shape, &input_dynamic_shape, normalized_dims, eps)?;
        let raw_node: NodeType = ln.into();

        self.create_node_inner(raw_node, name, "layer_norm", vec![input])
    }

    /// 创建 RMSNormOp 节点
    ///
    /// RMS 归一化核心计算（不含 gamma）
    pub fn create_rms_norm_op_node(
        &mut self,
        input: Rc<NodeInner>,
        normalized_dims: usize,
        eps: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::RMSNormOp;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let rn = RMSNormOp::new(&input_shape, &input_dynamic_shape, normalized_dims, eps)?;
        let raw_node: NodeType = rn.into();

        self.create_node_inner(raw_node, name, "rms_norm", vec![input])
    }

    /// 创建 Repeat 节点
    ///
    /// 沿各维度重复张量
    pub fn create_repeat_node(
        &mut self,
        input: Rc<NodeInner>,
        repeats: Vec<usize>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Repeat;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let repeat = Repeat::new(&input_shape, &input_dynamic_shape, repeats)?;
        let raw_node: NodeType = repeat.into();

        self.create_node_inner(raw_node, name, "repeat", vec![input])
    }

    /// 创建 Pad 节点
    ///
    /// 常量值填充: y = pad(x, paddings, value)
    pub fn create_pad_node(
        &mut self,
        input: Rc<NodeInner>,
        paddings: Vec<(usize, usize)>,
        pad_value: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Pad;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let pad = Pad::new(&input_shape, &input_dynamic_shape, paddings, pad_value)?;
        let raw_node: NodeType = pad.into();

        self.create_node_inner(raw_node, name, "pad", vec![input])
    }

    /// 创建 Pow 节点
    ///
    /// 逐元素幂运算: y = x^p
    pub fn create_pow_node(
        &mut self,
        input: Rc<NodeInner>,
        exponent: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Pow;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let pow = Pow::new(&input_shape, &input_dynamic_shape, exponent)?;
        let raw_node: NodeType = pow.into();

        self.create_node_inner(raw_node, name, "pow", vec![input])
    }

    /// 创建 Clip 节点
    ///
    /// 逐元素值域裁剪: y = clip(x, min, max)
    pub fn create_clip_node(
        &mut self,
        input: Rc<NodeInner>,
        min: f32,
        max: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Clip;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let clip = Clip::new(&input_shape, &input_dynamic_shape, min, max)?;
        let raw_node: NodeType = clip.into();

        self.create_node_inner(raw_node, name, "clip", vec![input])
    }

    /// 创建 WhereCond 节点
    ///
    /// 条件选择：`output = where(condition, x, y)`
    /// condition 是固定 Tensor（不参与梯度），x 和 y 是 Var 父节点。
    pub fn create_where_cond_node(
        &mut self,
        x: Rc<NodeInner>,
        y: Rc<NodeInner>,
        condition: crate::tensor::Tensor,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::WhereCond;

        let x_shape = x.shape();
        let y_shape = y.shape();
        let x_dynamic_shape = x.dynamic_shape();

        // 验证 x 和 y 形状一致
        if x_shape != y_shape {
            return Err(GraphError::ShapeMismatch {
                expected: x_shape.clone(),
                got: y_shape,
                message: "WhereCond: x 和 y 的形状必须一致".to_string(),
            });
        }

        let where_cond = WhereCond::new(&x_shape, &x_dynamic_shape, condition)?;
        let raw_node: NodeType = where_cond.into();

        self.create_node_inner(raw_node, name, "where_cond", vec![x, y])
    }

    /// 创建 SortNode 节点
    ///
    /// 沿指定轴排序（可微），返回排序后的值。
    /// 内部缓存排序索引，反向传播时通过逆置换 scatter 梯度。
    ///
    /// # 参数
    /// - `parent`: 输入节点
    /// - `axis`: 排序轴
    /// - `descending`: 是否降序
    pub fn create_sort_node(
        &mut self,
        parent: Rc<NodeInner>,
        axis: usize,
        descending: bool,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::SortNode;

        let parent_shape = parent.shape();
        let parent_dynamic_shape = parent.dynamic_shape();

        let sort_node = SortNode::new(&parent_shape, &parent_dynamic_shape, axis, descending)?;
        let raw_node: NodeType = sort_node.into();

        self.create_node_inner(raw_node, name, "sort", vec![parent])
    }

    /// 创建 Exp 节点
    ///
    /// 逐元素指数函数: y = e^x
    pub fn create_exp_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Exp;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let exp = Exp::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = exp.into();

        self.create_node_inner(raw_node, name, "exp", vec![input])
    }

    /// 创建 Square 节点
    ///
    /// 逐元素平方: y = x²
    pub fn create_square_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Square;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let square = Square::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = square.into();

        self.create_node_inner(raw_node, name, "square", vec![input])
    }

    /// 创建 Reciprocal 节点
    ///
    /// 逐元素倒数: y = 1/x
    pub fn create_reciprocal_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Reciprocal;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let reciprocal = Reciprocal::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = reciprocal.into();

        self.create_node_inner(raw_node, name, "reciprocal", vec![input])
    }

    /// 创建 Log10 节点
    ///
    /// 逐元素以 10 为底的对数: y = log10(x)
    pub fn create_log10_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Log10;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let log10 = Log10::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = log10.into();

        self.create_node_inner(raw_node, name, "log10", vec![input])
    }

    /// 创建 Log2 节点
    ///
    /// 逐元素以 2 为底的对数: y = log2(x)
    pub fn create_log2_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::Log2;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let log2 = Log2::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = log2.into();

        self.create_node_inner(raw_node, name, "log2", vec![input])
    }

    /// 创建 ReLU6 节点
    ///
    /// ReLU6 激活: y = min(max(0, x), 6)
    pub fn create_relu6_node(
        &mut self,
        input: Rc<NodeInner>,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::ReLU6;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let relu6 = ReLU6::new(&input_shape, &input_dynamic_shape)?;
        let raw_node: NodeType = relu6.into();

        self.create_node_inner(raw_node, name, "relu6", vec![input])
    }

    /// 创建 HardTanh 节点
    ///
    /// HardTanh 激活: y = min(max(min_val, x), max_val)
    pub fn create_hard_tanh_node(
        &mut self,
        input: Rc<NodeInner>,
        min_val: f32,
        max_val: f32,
        name: Option<&str>,
    ) -> Result<Rc<NodeInner>, GraphError> {
        use crate::nn::nodes::raw_node::HardTanh;

        let input_shape = input.shape();
        let input_dynamic_shape = input.dynamic_shape();

        let hard_tanh = HardTanh::new(&input_shape, &input_dynamic_shape, min_val, max_val)?;
        let raw_node: NodeType = hard_tanh.into();

        self.create_node_inner(raw_node, name, "hard_tanh", vec![input])
    }
}
