/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner 可视化相关
 *
 * 说明：
 * - 旧的 register_model_group / register_layer_group 已删除
 * - 层/模型/分布的 cluster 分组统一由 NodeGroupTag（NodeGroupContext RAII guard）驱动
 * - 可视化渲染通过 Var::snapshot_to_dot() / Var::to_dot() 实现
 */
