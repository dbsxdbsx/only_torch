/*
 * @Author       : 老董
 * @Date         : 2024-02-04 20:37:13
 * @LastEditors  : 老董
 * @LastEditTime : 2025-01-04 19:47:51
 * @Description  : 神经网络节点的显示格式化
 */

use super::NodeId;

/// 格式化神经网络节点的显示
///
/// # Arguments
/// * `id` - 节点ID
/// * `name` - 节点名称
/// * `type_name` - 节点类型名称
///
/// # Returns
/// 返回格式化后的字符串，格式为：`节点[id={}, name={}, type={}]`
pub(in crate::nn) fn format_node_display(id: NodeId, name: &str, type_name: &str) -> String {
    format!("节点[id={}, name={}, type={}]", id, name, type_name)
}
