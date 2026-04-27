/*
 * @Author       : 老董
 * @Date         : 2026-04-18
 * @Description  : ONNX 导入流水线：.onnx → GraphDescriptor + 权重
 *
 * 四层流水线：
 * 1. 解析层：读取 .onnx 二进制 → onnx_rs::ast::Model
 * 2. 符号表层：为每个 tensor name 分配唯一 u64 ID
 * 3. 算子映射层：ONNX Node → NodeTypeDescriptor
 * 4. 装配层：组装 GraphDescriptor + 权重 HashMap
 *
 * 模块组织（2026-04 拆分自单文件 ~1000 行）：
 * - mod.rs           : 入口 + 公开类型 (load_onnx, OnnxImportResult, ImportReport)
 * - assemble.rs      : 装配主循环 + Conv+bias / Gemm 拆分
 * - const_table.rs   : Constant + initializer 收集 + 元信息消费标记
 * - fold_reshape.rs  : Reshape 常量折叠
 * - fold_resize.rs   : Resize/Upsample 常量折叠
 * - split_narrow.rs  : Split → N×Narrow 重写
 * - util.rs          : SymbolTable + 共享辅助函数
 */

use std::path::Path;

use crate::nn::graph::onnx_error::OnnxError;

mod assemble;
mod const_table;
mod fold_reshape;
mod fold_resize;
mod split_narrow;
mod util;

pub use assemble::OnnxImportResult;

use util::{SymbolTable, validate_opset};

/// ONNX 导入过程的可观测报告（最小骨架版）
///
/// 当前仅含：
/// - `rewritten`：所有命中的模式重写记录（如 Conv+bias 拆分、Split→Narrow 等）
/// - `warnings`：非致命警告（如 bias 自动升维、Gemm 转置 B 等）
///
/// **范围控制**（参见 `chinese_chess_yolo_example_b4f3a201.plan.md` §3.4）：
/// 不含 `folded`/`shape_inference`/`provenance`/`origin_onnx_nodes` 等扩展字段，
/// 等真正撞到对应需求时再补，避免范围蔓延。
#[derive(Debug, Default, Clone)]
pub struct ImportReport {
    /// 已应用的模式重写记录（按命中顺序排列）
    pub rewritten: Vec<RewriteRecord>,
    /// 非致命警告
    pub warnings: Vec<String>,
}

/// 单条 pattern rewrite 记录
///
/// 描述一次 ONNX → only_torch 节点重写的输入/输出对应关系，
/// 便于上层调试"为什么 ONNX 节点数和 only_torch 节点数不一致"。
#[derive(Debug, Clone)]
pub struct RewriteRecord {
    /// 模式名（如 `"conv_with_bias_to_conv_plus_add"`、`"split_to_narrows"`、
    /// `"constant_fold_into_reshape"`）
    pub pattern: &'static str,
    /// 该重写"消化"了哪些 ONNX 原始节点（按 ONNX `node.name` 收集）
    pub consumed_onnx_nodes: Vec<String>,
    /// 该重写在 only_torch 内"产出"了哪些 descriptor 节点 ID
    pub produced_descriptor_nodes: Vec<u64>,
}

/// 从 .onnx 文件加载为 GraphDescriptor + 权重
pub fn load_onnx<P: AsRef<Path>>(path: P) -> Result<OnnxImportResult, OnnxError> {
    let bytes = std::fs::read(path)?;
    load_onnx_from_bytes(&bytes)
}

/// 从内存中的 .onnx 字节流加载
pub fn load_onnx_from_bytes(bytes: &[u8]) -> Result<OnnxImportResult, OnnxError> {
    // ── 第 1 层：解析 ──
    let model = onnx_rs::parse(bytes).map_err(|e| OnnxError::ParseError(format!("{e}")))?;

    // 验证 opset 版本
    validate_opset(&model)?;

    let graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::InvalidGraph("模型不含计算图".to_string()))?;

    // ── 第 2 层：符号表 ──
    let mut symbols = SymbolTable::new();
    symbols.register_graph(graph);

    // ── 第 3 & 4 层：算子映射 + 装配 ──
    assemble::assemble(graph, &mut symbols)
}
