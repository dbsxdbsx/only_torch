/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化模块
 *
 * Genome-centric 层级进化：
 * - gene.rs: 基因数据结构（LayerGene, NetworkGenome, LayerConfig 等）
 * - mutation.rs: 变异操作（Mutation trait + MutationRegistry）
 * - builder.rs: Genome → Graph 转换 + Lamarckian 权重继承
 *
 * - convergence.rs: 训练收敛检测（ConvergenceDetector）
 *
 * 后续阶段将扩展：
 * - callback.rs: 回调接口（Phase 6）
 */

pub mod builder;
pub mod convergence;
pub mod gene;
pub mod mutation;
pub mod task;

#[cfg(test)]
mod tests;
