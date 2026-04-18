/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner describe/summary 相关方法
 *
 * GraphInner 不再提供 describe()/summary()：原实现依赖已移除的 nodes 表。
 * 图描述与摘要应通过 Var 遍历（例如 Var::to_dot()）完成。
 * TODO: 如需要，可在此实现基于 parameters 注册表的简化版描述
 */

use super::GraphInner;

impl GraphInner {
    // ========== 图描述（describe）==========
    // 原方法已移除，新的描述功能应通过 Var.to_dot() 实现

    // ========== 模型摘要（summary）==========
    // 原方法已移除，可通过 Var 可视化替代
}
