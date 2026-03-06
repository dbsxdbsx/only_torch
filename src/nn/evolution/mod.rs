/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化模块
 *
 * Genome-centric 层级进化：
 * - gene.rs: 基因数据结构（LayerGene, NetworkGenome, LayerConfig 等）
 *
 * 后续阶段将扩展：
 * - mutation.rs: 变异操作（Phase 2）
 * - builder.rs: Genome → Graph 转换（Phase 3）
 * - convergence.rs: 收敛检测（Phase 4）
 * - task.rs: 演化任务 trait（Phase 5）
 * - callback.rs: 回调接口（Phase 6）
 */

pub mod gene;

#[cfg(test)]
mod tests;
