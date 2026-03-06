/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : Genome → Graph 转换 + Lamarckian 权重继承
 *
 * build() 按 resolve_dimensions() 顺序逐层创建计算图，
 * capture_weights() / restore_weights() 实现跨代权重复用。
 */

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::Rng;

use crate::nn::{Graph, GraphError, Linear, Var, VarActivationOps};
use crate::tensor::Tensor;

use super::gene::{ActivationType, LayerConfig, NetworkGenome};

// ==================== BuildResult ====================

/// build() 的完整返回值
///
/// 将构建过程中产生的所有信息一次性交付给调用者，
/// 避免事后从 Graph 中重新收集。
pub struct BuildResult {
    /// 输入节点（用于设置训练/测试数据）
    pub input: Var,
    /// 输出节点（用于获取预测结果）
    pub output: Var,
    /// innovation_number → 该层的参数变量列表（如 Linear 有 [W, b]）
    ///
    /// capture_weights() / restore_weights() 按 innovation_number 匹配参数。
    /// Optimizer 所需的扁平参数列表通过 all_parameters() 派生。
    pub layer_params: HashMap<u64, Vec<Var>>,
    /// 内部引用的 Graph（保持 Graph 存活，防止 Var 中的 Weak 失效）
    pub graph: Graph,
}

impl BuildResult {
    /// 所有可训练参数的扁平列表（用于创建 Optimizer）
    ///
    /// 从 layer_params 派生，确保与 capture/restore 使用同一数据源。
    /// 按 innovation_number 排序以保证确定性顺序（HashMap 迭代顺序不确定）。
    pub fn all_parameters(&self) -> Vec<Var> {
        let mut keys: Vec<_> = self.layer_params.keys().copied().collect();
        keys.sort_unstable();
        keys.iter()
            .flat_map(|k| self.layer_params[k].iter().cloned())
            .collect()
    }
}

// ==================== InheritReport ====================

/// 权重继承报告
pub struct InheritReport {
    /// 成功继承的参数张量数
    pub inherited: usize,
    /// 保留初始化值的参数张量数（新层或形状变化）
    pub reinitialized: usize,
}

// ==================== 内部辅助 ====================

fn apply_activation(var: &Var, act_type: &ActivationType) -> Var {
    match act_type {
        ActivationType::ReLU => var.relu(),
        ActivationType::LeakyReLU { alpha } => var.leaky_relu(*alpha),
        ActivationType::Tanh => var.tanh(),
        ActivationType::Sigmoid => var.sigmoid(),
        ActivationType::GELU => var.gelu(),
        ActivationType::SiLU => var.silu(),
        ActivationType::Softplus => var.softplus(),
        ActivationType::ReLU6 => var.relu6(),
    }
}

// ==================== NetworkGenome 构建与权重管理 ====================

impl NetworkGenome {
    /// 从基因组构建计算图
    ///
    /// 内部调用 resolve_dimensions() 推导维度链，逐层创建 Layer 并收集参数。
    /// 遇到 skip_edge 的目标层时，自动在其输入处生成聚合操作（Phase 7B）。
    ///
    /// rng 用于派生 Graph seed，确保参数初始化受 Evolution seed 控制。
    pub fn build(&self, rng: &mut StdRng) -> Result<BuildResult, GraphError> {
        let resolved = self
            .resolve_dimensions()
            .map_err(|e| GraphError::ComputationError(e.to_string()))?;

        let graph_seed: u64 = rng.r#gen();
        let graph = Graph::new_with_seed(graph_seed);

        let input = graph.input_shape(&[1, self.input_dim], Some("evo_input"))?;
        let mut current = input.clone();
        let mut layer_params: HashMap<u64, Vec<Var>> = HashMap::new();

        for dim in &resolved {
            let layer = self
                .layers
                .iter()
                .find(|l| l.innovation_number == dim.innovation_number && l.enabled)
                .expect("resolve_dimensions 返回的创新号必须对应一个启用的层");

            match &layer.layer_config {
                LayerConfig::Linear { out_features } => {
                    let name = format!("evo_L{}", layer.innovation_number);
                    let linear =
                        Linear::new(&graph, dim.in_dim, *out_features, true, &name)?;
                    current = linear.forward(&current);

                    let mut params = vec![linear.weights().clone()];
                    if let Some(bias) = linear.bias() {
                        params.push(bias.clone());
                    }
                    layer_params.insert(layer.innovation_number, params);
                }
                LayerConfig::Activation { activation_type } => {
                    current = apply_activation(&current, activation_type);
                }
                LayerConfig::Dropout { .. }
                | LayerConfig::Rnn { .. }
                | LayerConfig::Lstm { .. }
                | LayerConfig::Gru { .. } => {
                    return Err(GraphError::ComputationError(format!(
                        "build() 尚未支持 {} 层类型（层 innovation={}）",
                        layer.layer_config, layer.innovation_number
                    )));
                }
            }
        }

        Ok(BuildResult {
            input,
            output: current,
            layer_params,
            graph,
        })
    }

    /// 将当前计算图的权重捕获到 Genome 的 weight_snapshots 中
    ///
    /// 训练完成后调用。按 innovation_number 索引，
    /// 每层的参数张量列表保存为 Vec<Tensor>（如 Linear 保存 [W, b]）。
    ///
    /// 注意：此方法是全量替换（非 merge），不在当前 build 中的层（如 disabled）
    /// 的旧快照会被丢弃。Phase 7B 引入 disable/enable 语义时需评估是否改为 merge。
    pub fn capture_weights(&mut self, build: &BuildResult) -> Result<(), GraphError> {
        let mut snapshots: HashMap<u64, Vec<Tensor>> = HashMap::new();
        for (&inn, params) in &build.layer_params {
            let mut tensors = Vec::new();
            for param in params {
                let tensor = param.value()?.ok_or_else(|| {
                    GraphError::ComputationError(format!("层 {inn} 的参数无值"))
                })?;
                tensors.push(tensor);
            }
            snapshots.insert(inn, tensors);
        }
        self.set_weight_snapshots(snapshots);
        Ok(())
    }

    /// 从 weight_snapshots 恢复权重到当前计算图
    ///
    /// build() 之后、训练之前调用。
    /// 按 innovation_number 匹配：相同创新号且形状相同的参数直接复制，
    /// 形状不匹配或无快照的参数保留初始化值。
    pub fn restore_weights(&self, build: &BuildResult) -> Result<InheritReport, GraphError> {
        let mut inherited = 0usize;
        let mut reinitialized = 0usize;

        for (&inn, params) in &build.layer_params {
            if let Some(snapshots) = self.weight_snapshots().get(&inn) {
                for (i, param) in params.iter().enumerate() {
                    if let Some(snapshot) = snapshots.get(i) {
                        let current_val = param.value()?;
                        let shapes_match = current_val
                            .as_ref()
                            .map(|t| t.shape() == snapshot.shape())
                            .unwrap_or(false);

                        if shapes_match {
                            param.set_value(snapshot)?;
                            inherited += 1;
                        } else {
                            reinitialized += 1;
                        }
                    } else {
                        reinitialized += 1;
                    }
                }
            } else {
                reinitialized += params.len();
            }
        }

        Ok(InheritReport {
            inherited,
            reinitialized,
        })
    }
}
