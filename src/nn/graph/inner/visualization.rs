/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 可视化相关（Phase 3 简化版）
 *
 * Phase 3 清理说明：
 * - 原有的 to_dot()/save_visualization() 方法依赖 nodes HashMap，已移除
 * - 新的可视化功能通过 Var::to_dot() / Var::save_visualization() 实现
 * - 此文件仅保留层分组注册等辅助功能
 */

use super::super::types::{GroupKind, LayerGroup};
use super::GraphInner;
use crate::nn::NodeId;

impl GraphInner {
    // ========== 层分组管理 ==========

    /// 注册模型分组（供 Layer 调用）
    ///
    /// 用于可视化时将同一模型的节点用半透明框分组显示。
    /// Layer 创建时自动调用此方法。
    ///
    /// # 参数
    /// - `name`: 分组名称（如 "encoder"、"decoder"）
    /// - `node_ids`: 属于该分组的节点 ID 列表
    pub fn register_model_group(&mut self, name: String, node_ids: Vec<NodeId>) {
        // 如果该分组已存在，合并节点；否则创建新分组
        if let Some(existing) = self
            .layer_groups
            .iter_mut()
            .find(|g| g.name == name && matches!(g.kind, GroupKind::Model))
        {
            // 合并节点（去重）
            for id in node_ids {
                if !existing.node_ids.contains(&id) {
                    existing.node_ids.push(id);
                }
            }
        } else {
            self.layer_groups.push(LayerGroup {
                name,
                layer_type: "Model".to_string(),
                description: String::new(),
                node_ids,
                kind: GroupKind::Model,
                recurrent_steps: None,
                min_steps: None,
                max_steps: None,
                hidden_node_ids: vec![],
                folded_nodes: vec![],
                output_proxy: None,
            });
        }
    }
}
