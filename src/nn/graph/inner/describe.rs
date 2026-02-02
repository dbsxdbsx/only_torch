/*
 * @Author       : 老董
 * @Date         : 2026-01-27
 * @Description  : GraphInner describe/summary 相关方法
 *
 * Phase 3 清理说明：
 * - 原有的 describe()/summary() 方法依赖 nodes HashMap，已被移除
 * - 新的描述功能应通过 Var 遍历实现
 * - TODO: 如需要，可在此实现基于 parameters 注册表的简化版描述
 */

use super::GraphInner;

impl GraphInner {
    // ========== 图描述（describe）==========
    // 原方法已移除，新的描述功能应通过 Var.to_dot() 实现

    // ========== 模型摘要（summary）==========
    // 原方法已移除，可通过 Var 可视化替代
}
