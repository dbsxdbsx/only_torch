/*
 * @Author       : 老董
 * @Date         : 2026-03-11
 * @Description  : 统一 .otm 模型 save/load 综合测试
 *
 * 覆盖场景：
 * A. 基本往返：MLP / CNN+BN / Dropout / skip 连接 / 多输出
 * B. 演化兼容性：EvolutionResult save/load 使用新 v2 格式
 * C. 跨模型交互：手动模型 ↔ 演化模型的互操作
 * D. 错误处理：无效文件、不存在的文件、版本不匹配、截断文件
 * E. 边界情况：嵌套目录、仅输入无参数、加载后继续训练
 */

use crate::nn::{
    BatchNorm, Conv2d, Graph, Linear, RebuildResult, Var,
    VarActivationOps, VarLossOps, VarRegularizationOps, VarShapeOps,
};
use crate::nn::optimizer::{Optimizer, SGD};
use crate::tensor::Tensor;

// ==================== 辅助函数 ====================

/// 比较两个 f32 切片，允许一定误差
fn assert_vec_close(a: &[f32], b: &[f32], eps: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{msg}: 长度不匹配 {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < eps,
            "{msg}: 第 {i} 个元素不匹配: {x} vs {y}"
        );
    }
}

/// 喂入数据并前向传播，返回第一个输出的 f32 数据
fn predict(result: &RebuildResult, data: &Tensor) -> Vec<f32> {
    result.inputs[0].1.set_value(data).expect("set_value 失败");
    result.graph.forward(&result.outputs[0]).expect("forward 失败");
    result.outputs[0].value().expect("value 失败").unwrap().to_vec()
}

// ==================== A. 基本往返测试 ====================

/// MLP 往返：Linear → ReLU → Linear
#[test]
fn test_save_load_mlp_roundtrip() {
    let path = "test_otm_mlp_roundtrip";

    // 1. 构建模型并前向传播
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4])).unwrap();
    let fc1 = Linear::new(&graph, 4, 8, true, "fc1").unwrap();
    let fc2 = Linear::new(&graph, 8, 2, true, "fc2").unwrap();
    let h = fc1.forward(&x).relu();
    let out = fc2.forward(&h);

    graph.forward(&out).unwrap();
    let original_pred = out.value().unwrap().unwrap().to_vec();

    // 2. 保存
    graph.save_model(path, &[&out]).unwrap();
    assert!(std::path::Path::new("test_otm_mlp_roundtrip.otm").exists());

    // 3. 加载
    let loaded = Graph::load_model(path).unwrap();

    // 4. 验证结构
    assert_eq!(loaded.inputs.len(), 1, "应有 1 个输入");
    assert_eq!(loaded.outputs.len(), 1, "应有 1 个输出");

    // 5. 验证预测一致
    let loaded_pred = predict(&loaded, &Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]));
    assert_vec_close(&original_pred, &loaded_pred, 1e-5, "MLP 预测");

    // 6. 验证 eval 模式
    assert!(loaded.graph.is_eval(), "加载后应为 eval 模式");

    // 清理
    std::fs::remove_file("test_otm_mlp_roundtrip.otm").ok();
}

/// CNN + BatchNorm 往返：Conv2d → BN → ReLU → Flatten → Linear
#[test]
fn test_save_load_cnn_bn_roundtrip() {
    let path = "test_otm_cnn_bn_roundtrip";

    // 1. 构建 CNN 模型
    let graph = Graph::new_with_seed(42);
    // 输入: [1, 1, 6, 6] (batch=1, channels=1, 6x6)
    let x = graph
        .input(&Tensor::normal(0.0, 1.0, &[1, 1, 6, 6]))
        .unwrap();
    let conv = Conv2d::new(&graph, 1, 4, (3, 3), (1, 1), (0, 0), (1, 1), true, "conv1").unwrap();
    let bn = BatchNorm::new(&graph, 4, 1e-5, 0.1, "bn1").unwrap();

    let h = conv.forward(&x);
    let h = bn.forward(&h).relu();
    let h = h.flatten().unwrap(); // [1, 4*4*4] = [1, 64]
    let fc = Linear::new(&graph, 64, 2, true, "fc_out").unwrap();
    let out = fc.forward(&h);

    graph.eval(); // BN 使用 running stats
    graph.forward(&out).unwrap();
    let original_pred = out.value().unwrap().unwrap().to_vec();

    // 2. 保存
    graph.save_model(path, &[&out]).unwrap();

    // 3. 加载并验证
    let loaded = Graph::load_model(path).unwrap();
    // 设置与原始相同的输入
    let loaded_pred = predict(&loaded, &x.value().unwrap().unwrap());
    assert_vec_close(&original_pred, &loaded_pred, 1e-4, "CNN+BN 预测");

    // 清理
    std::fs::remove_file("test_otm_cnn_bn_roundtrip.otm").ok();
}

/// Dropout 模型往返：加载后 eval 模式，dropout 应直接通过
#[test]
fn test_save_load_dropout_eval() {
    let path = "test_otm_dropout_eval";

    // 1. 构建带 dropout 的模型
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 4])).unwrap();
    let fc = Linear::new(&graph, 4, 4, true, "fc").unwrap();
    let h = fc.forward(&x).dropout(0.5).unwrap();

    // eval 模式下 dropout 不生效
    graph.eval();
    graph.forward(&h).unwrap();
    let pred1 = h.value().unwrap().unwrap().to_vec();

    // 再次 forward，eval 模式下应完全相同
    graph.forward(&h).unwrap();
    let pred2 = h.value().unwrap().unwrap().to_vec();
    assert_vec_close(&pred1, &pred2, 1e-6, "eval 模式 dropout 应确定性");

    // 2. 保存 & 加载
    graph.save_model(path, &[&h]).unwrap();
    let loaded = Graph::load_model(path).unwrap();

    // 3. 验证加载后为 eval 模式，结果一致
    assert!(loaded.graph.is_eval());
    let loaded_pred = predict(&loaded, &Tensor::ones(&[1, 4]));
    assert_vec_close(&pred1, &loaded_pred, 1e-5, "Dropout eval 预测");

    // 清理
    std::fs::remove_file("test_otm_dropout_eval.otm").ok();
}

/// Skip 连接（残差）往返：y = relu(fc1(x)) + x → fc2
#[test]
fn test_save_load_skip_connection() {
    let path = "test_otm_skip_conn";

    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::new(&[0.5, -0.3, 0.8, 0.1], &[1, 4])).unwrap();

    let fc1 = Linear::new(&graph, 4, 4, true, "fc1").unwrap();
    let h = fc1.forward(&x).relu();
    let skip = &h + &x; // 残差连接

    let fc2 = Linear::new(&graph, 4, 2, true, "fc2").unwrap();
    let out = fc2.forward(&skip);

    graph.forward(&out).unwrap();
    let original_pred = out.value().unwrap().unwrap().to_vec();

    graph.save_model(path, &[&out]).unwrap();
    let loaded = Graph::load_model(path).unwrap();

    let loaded_pred = predict(&loaded, &Tensor::new(&[0.5, -0.3, 0.8, 0.1], &[1, 4]));
    assert_vec_close(&original_pred, &loaded_pred, 1e-5, "Skip 连接预测");

    // 清理
    std::fs::remove_file("test_otm_skip_conn.otm").ok();
}

/// 多输出模型往返
#[test]
fn test_save_load_multi_output() {
    let path = "test_otm_multi_output";

    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])).unwrap();

    let fc_shared = Linear::new(&graph, 3, 8, true, "shared").unwrap();
    let h = fc_shared.forward(&x).relu();

    // 分支 1：分类头
    let fc_cls = Linear::new(&graph, 8, 3, true, "cls_head").unwrap();
    let cls_out = fc_cls.forward(&h);

    // 分支 2：回归头
    let fc_reg = Linear::new(&graph, 8, 1, true, "reg_head").unwrap();
    let reg_out = fc_reg.forward(&h);

    graph.forward(&cls_out).unwrap();
    graph.forward(&reg_out).unwrap();
    let original_cls = cls_out.value().unwrap().unwrap().to_vec();
    let original_reg = reg_out.value().unwrap().unwrap().to_vec();

    // 保存两个输出
    graph.save_model(path, &[&cls_out, &reg_out]).unwrap();

    let loaded = Graph::load_model(path).unwrap();
    assert_eq!(loaded.inputs.len(), 1, "应有 1 个输入");
    assert_eq!(loaded.outputs.len(), 2, "应有 2 个输出");

    // 喂入数据，分别 forward 两个输出
    loaded.inputs[0].1.set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3])).unwrap();

    loaded.graph.forward(&loaded.outputs[0]).unwrap();
    let loaded_cls = loaded.outputs[0].value().unwrap().unwrap().to_vec();

    loaded.graph.forward(&loaded.outputs[1]).unwrap();
    let loaded_reg = loaded.outputs[1].value().unwrap().unwrap().to_vec();

    assert_vec_close(&original_cls, &loaded_cls, 1e-5, "分类头预测");
    assert_vec_close(&original_reg, &loaded_reg, 1e-5, "回归头预测");

    // 清理
    std::fs::remove_file("test_otm_multi_output.otm").ok();
}

// ==================== B. 演化兼容性测试 ====================

/// 演化模型 save/load 往返（v2 格式）
#[test]
fn test_evolution_save_load_v2_roundtrip() {
    use crate::nn::evolution::gene::TaskMetric;
    use crate::nn::evolution::{Evolution, EvolutionResult};

    let path = "test_otm_evo_v2_roundtrip";

    let train = (
        vec![
            Tensor::new(&[0.0, 0.0], &[2]),
            Tensor::new(&[0.0, 1.0], &[2]),
            Tensor::new(&[1.0, 0.0], &[2]),
            Tensor::new(&[1.0, 1.0], &[2]),
        ],
        vec![
            Tensor::new(&[0.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[0.0], &[1]),
        ],
    );

    let result = Evolution::supervised(train.clone(), train, TaskMetric::Accuracy)
        .with_target_metric(0.75)
        .with_seed(42)
        .with_verbose(false)
        .run()
        .expect("演化失败");

    let test_input = Tensor::new(&[1.0, 0.0], &[2]);
    let original_pred = result.predict(&test_input).unwrap().to_vec();

    result.save(path).unwrap();
    assert!(std::path::Path::new("test_otm_evo_v2_roundtrip.otm").exists());

    let loaded = EvolutionResult::load(path).unwrap();
    let loaded_pred = loaded.predict(&test_input).unwrap().to_vec();

    assert_vec_close(&original_pred, &loaded_pred, 1e-5, "演化模型预测");
    assert_eq!(loaded.generations, result.generations, "代数应一致");

    // 清理
    std::fs::remove_file("test_otm_evo_v2_roundtrip.otm").ok();
}

// ==================== C. 跨模型交互测试 ====================

/// 演化模型 .otm → Graph::load_model 加载（提取纯推理网络）
#[test]
fn test_evolution_otm_loaded_as_manual() {
    use crate::nn::evolution::gene::TaskMetric;
    use crate::nn::evolution::Evolution;

    let path = "test_otm_evo_to_manual";

    let train = (
        vec![
            Tensor::new(&[0.0, 0.0], &[2]),
            Tensor::new(&[0.0, 1.0], &[2]),
            Tensor::new(&[1.0, 0.0], &[2]),
            Tensor::new(&[1.0, 1.0], &[2]),
        ],
        vec![
            Tensor::new(&[0.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[0.0], &[1]),
        ],
    );

    let result = Evolution::supervised(train.clone(), train, TaskMetric::Accuracy)
        .with_target_metric(0.75)
        .with_seed(42)
        .with_verbose(false)
        .run()
        .expect("演化失败");

    let test_input = Tensor::new(&[1.0, 0.0], &[2]);
    let original_pred = result.predict(&test_input).unwrap().to_vec();

    result.save(path).unwrap();

    // 用 Graph::load_model 加载（忽略 evolution 元数据，仅重建图）
    let loaded = Graph::load_model(path).unwrap();
    assert!(!loaded.inputs.is_empty(), "应有输入节点");
    assert!(!loaded.outputs.is_empty(), "应有输出节点");

    // 演化模型期望带 batch 维度的输入 [1, 2]（EvolutionResult::predict 自动添加）
    let batched_input = Tensor::new(&[1.0, 0.0], &[1, 2]);
    let loaded_pred = predict(&loaded, &batched_input);
    assert_vec_close(&original_pred, &loaded_pred, 1e-5, "演化→手动加载预测");

    // 清理
    std::fs::remove_file("test_otm_evo_to_manual.otm").ok();
}

/// 手动模型 .otm → EvolutionResult::load 应失败（缺少 evolution 字段）
#[test]
fn test_manual_otm_to_evolution_load_fails() {
    use crate::nn::evolution::EvolutionResult;

    let path = "test_otm_manual_to_evo";

    // 构建并保存手动模型
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 2])).unwrap();
    let fc = Linear::new(&graph, 2, 1, true, "fc").unwrap();
    let out = fc.forward(&x);
    graph.save_model(path, &[&out]).unwrap();

    // 尝试作为演化模型加载 → 应报错
    let result = EvolutionResult::load(path);
    assert!(result.is_err(), "手动模型不应被 EvolutionResult::load 加载");
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => unreachable!(),
    };
    assert!(
        err_msg.contains("evolution"),
        "错误信息应提到 evolution 字段缺失: {}",
        err_msg
    );

    // 清理
    std::fs::remove_file("test_otm_manual_to_evo.otm").ok();
}

/// 手动模型保存 → 加载 → 用 optimizer 继续训练（验证 forward/backward/step 全链路）
#[test]
fn test_load_then_continue_training() {
    let path = "test_otm_continue_train";

    // 1. 构建并保存
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();
    let fc = Linear::new(&graph, 2, 1, true, "fc").unwrap();
    let out = fc.forward(&x).sigmoid();
    graph.save_model(path, &[&out]).unwrap();

    // 2. 加载
    let loaded = Graph::load_model(path).unwrap();

    // 3. 准备训练
    loaded.graph.train();

    // 从参数注册表获取 Var 列表（供 optimizer 使用）
    let graph_rc = loaded.graph.inner_rc();
    let param_vars: Vec<Var> = loaded.graph.inner().get_all_parameters()
        .into_iter()
        .map(|(_, node)| Var::new_with_rc_graph(node, &graph_rc))
        .collect();
    assert!(!param_vars.is_empty(), "应有可训练参数");

    // 记录训练前的参数值
    let params_before: Vec<Vec<f32>> = param_vars.iter()
        .map(|p| p.node().value().unwrap().to_vec())
        .collect();

    // 创建 target 和 optimizer
    let target_var = loaded.graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let mut optimizer = SGD::new(&loaded.graph, &param_vars, 0.1);

    // 4. 跑 3 步训练
    let loss = loaded.outputs[0].mse_loss(&target_var).unwrap();
    for _ in 0..3 {
        loaded.inputs[0].1.set_value(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();
        target_var.set_value(&Tensor::new(&[1.0], &[1, 1])).unwrap();
        let loss_val = optimizer.minimize(&loss).unwrap();
        assert!(loss_val.is_finite(), "loss 应为有限值");
    }

    // 5. 验证参数确实被更新（至少有一个参数变化了）
    let params_after: Vec<Vec<f32>> = param_vars.iter()
        .map(|p| p.node().value().unwrap().to_vec())
        .collect();
    let any_changed = params_before.iter().zip(params_after.iter())
        .any(|(before, after)| before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-10));
    assert!(any_changed, "训练后至少一个参数应被更新");

    // 6. 训练后仍可正常推理
    loaded.graph.eval();
    loaded.inputs[0].1.set_value(&Tensor::new(&[0.0, 1.0], &[1, 2])).unwrap();
    loaded.graph.forward(&loaded.outputs[0]).unwrap();
    let pred = loaded.outputs[0].value().unwrap().unwrap();
    assert_eq!(pred.shape(), &[1, 1], "训练后推理输出形状应正确");

    // 清理
    std::fs::remove_file("test_otm_continue_train.otm").ok();
}

/// 演化模型 → Graph::load_model → 用 optimizer 手动继续训练
///
/// 关键场景：用户先通过演化搜索到一个好架构，然后用传统方式精调。
/// 验证 forward → backward → optimizer.step 全链路跨模型类型可用。
#[test]
fn test_evolution_model_load_and_manual_train() {
    use crate::nn::evolution::gene::TaskMetric;
    use crate::nn::evolution::Evolution;

    let path = "test_otm_evo_manual_train";

    // 1. 演化出一个 XOR 模型
    let train = (
        vec![
            Tensor::new(&[0.0, 0.0], &[2]),
            Tensor::new(&[0.0, 1.0], &[2]),
            Tensor::new(&[1.0, 0.0], &[2]),
            Tensor::new(&[1.0, 1.0], &[2]),
        ],
        vec![
            Tensor::new(&[0.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[1.0], &[1]),
            Tensor::new(&[0.0], &[1]),
        ],
    );
    let result = Evolution::supervised(train.clone(), train, TaskMetric::Accuracy)
        .with_target_metric(0.75)
        .with_seed(42)
        .with_verbose(false)
        .run()
        .expect("演化失败");
    result.save(path).unwrap();

    // 2. 用 Graph::load_model 加载（不依赖演化元数据）
    let loaded = Graph::load_model(path).unwrap();
    loaded.graph.train();

    // 3. 获取参数 Var，创建 optimizer
    let graph_rc = loaded.graph.inner_rc();
    let param_vars: Vec<Var> = loaded.graph.inner().get_all_parameters()
        .into_iter()
        .map(|(_, node)| Var::new_with_rc_graph(node, &graph_rc))
        .collect();
    assert!(!param_vars.is_empty(), "演化模型应有可训练参数");

    let params_before: Vec<Vec<f32>> = param_vars.iter()
        .map(|p| p.node().value().unwrap().to_vec())
        .collect();

    // 演化模型的输入需要 batch 维度 [batch, input_dim]
    let target_var = loaded.graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let mut optimizer = SGD::new(&loaded.graph, &param_vars, 0.01);

    // 4. 跑几步训练，验证全链路不报错
    let loss = loaded.outputs[0].mse_loss(&target_var).unwrap();
    for _ in 0..3 {
        loaded.inputs[0].1.set_value(&Tensor::new(&[1.0, 0.0], &[1, 2])).unwrap();
        target_var.set_value(&Tensor::new(&[1.0], &[1, 1])).unwrap();
        let loss_val = optimizer.minimize(&loss).unwrap();
        assert!(loss_val.is_finite(), "loss 应为有限值");
    }

    // 5. 验证参数确实被更新
    let params_after: Vec<Vec<f32>> = param_vars.iter()
        .map(|p| p.node().value().unwrap().to_vec())
        .collect();
    let any_changed = params_before.iter().zip(params_after.iter())
        .any(|(before, after)| before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-10));
    assert!(any_changed, "训练后至少一个参数应被更新");

    // 6. 训练后仍可推理
    loaded.graph.eval();
    loaded.inputs[0].1.set_value(&Tensor::new(&[0.0, 1.0], &[1, 2])).unwrap();
    loaded.graph.forward(&loaded.outputs[0]).unwrap();
    let pred = loaded.outputs[0].value().unwrap().unwrap();
    assert!(!pred.to_vec().is_empty(), "训练后应能正常推理");

    // 清理
    std::fs::remove_file("test_otm_evo_manual_train.otm").ok();
}

// ==================== D. 错误处理测试 ====================

/// 加载无效文件（魔数不匹配）
#[test]
fn test_load_model_invalid_magic() {
    let file_path = "test_otm_invalid_magic.otm";
    std::fs::write(file_path, b"NOT_A_VALID_OTM_FILE").unwrap();

    let result = Graph::load_model("test_otm_invalid_magic");
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => unreachable!(),
    };
    assert!(
        err_msg.contains("魔数不匹配"),
        "错误信息应提到魔数: {}",
        err_msg
    );

    std::fs::remove_file(file_path).ok();
}

/// 加载不存在的文件
#[test]
fn test_load_model_nonexistent() {
    let result = Graph::load_model("totally_nonexistent_model_file");
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => unreachable!(),
    };
    assert!(
        err_msg.contains("无法打开文件"),
        "错误信息应提到文件不存在: {}",
        err_msg
    );
}

/// 加载版本不匹配的文件
#[test]
fn test_load_model_version_mismatch() {
    let file_path = "test_otm_version_mismatch.otm";

    // 写入正确 magic 但错误版本
    let mut data = Vec::new();
    data.extend_from_slice(b"OTMD");
    data.extend_from_slice(&99u32.to_le_bytes()); // version = 99
    std::fs::write(file_path, &data).unwrap();

    let result = Graph::load_model("test_otm_version_mismatch");
    assert!(result.is_err());
    let err_msg = match result {
        Err(e) => format!("{e}"),
        Ok(_) => unreachable!(),
    };
    assert!(
        err_msg.contains("不支持的 .otm 版本"),
        "错误信息应提到版本: {}",
        err_msg
    );

    std::fs::remove_file(file_path).ok();
}

/// 加载截断的文件（magic 正确但数据不完整）
#[test]
fn test_load_model_truncated() {
    let file_path = "test_otm_truncated.otm";

    // 只写 magic + version，缺少后续数据
    let mut data = Vec::new();
    data.extend_from_slice(b"OTMD");
    data.extend_from_slice(&2u32.to_le_bytes());
    std::fs::write(file_path, &data).unwrap();

    let result = Graph::load_model("test_otm_truncated");
    assert!(result.is_err(), "截断文件应加载失败");

    std::fs::remove_file(file_path).ok();
}

// ==================== E. 边界情况测试 ====================

/// 保存到嵌套目录（自动创建）
#[test]
fn test_save_model_nested_directory() {
    let path = "test_otm_nested_dir_model/sub1/sub2/model";

    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 2])).unwrap();
    let fc = Linear::new(&graph, 2, 1, true, "fc").unwrap();
    let out = fc.forward(&x);

    graph.save_model(path, &[&out]).unwrap();
    assert!(
        std::path::Path::new("test_otm_nested_dir_model/sub1/sub2/model.otm").exists(),
        "文件应在嵌套目录中创建"
    );

    // 清理
    std::fs::remove_dir_all("test_otm_nested_dir_model").ok();
}

/// 不同 batch size 的输入
#[test]
fn test_save_load_different_batch() {
    let path = "test_otm_diff_batch";

    // 用 batch=1 构建并保存
    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::new(&[1.0, 2.0], &[1, 2])).unwrap();
    let fc = Linear::new(&graph, 2, 3, true, "fc").unwrap();
    let out = fc.forward(&x);

    graph.forward(&out).unwrap();
    let pred_b1 = out.value().unwrap().unwrap().to_vec();

    graph.save_model(path, &[&out]).unwrap();

    // 加载后用 batch=1 预测（应一致）
    let loaded = Graph::load_model(path).unwrap();
    let loaded_pred = predict(&loaded, &Tensor::new(&[1.0, 2.0], &[1, 2]));
    assert_vec_close(&pred_b1, &loaded_pred, 1e-5, "batch=1 预测");

    // 用 batch=3 预测（应正常工作）
    let batch_data = Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        &[3, 2],
    );
    loaded.inputs[0].1.set_value(&batch_data).unwrap();
    loaded.graph.forward(&loaded.outputs[0]).unwrap();
    let batch_result = loaded.outputs[0].value().unwrap().unwrap();
    assert_eq!(batch_result.shape(), &[3, 3], "batch=3 输出形状应为 [3, 3]");

    // 清理
    std::fs::remove_file("test_otm_diff_batch.otm").ok();
}

/// 保存并加载两次（确保文件可重复读取）
#[test]
fn test_load_model_twice() {
    let path = "test_otm_load_twice";

    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 3])).unwrap();
    let fc = Linear::new(&graph, 3, 2, true, "fc").unwrap();
    let out = fc.forward(&x);
    graph.save_model(path, &[&out]).unwrap();

    // 第一次加载
    let loaded1 = Graph::load_model(path).unwrap();
    let pred1 = predict(&loaded1, &Tensor::ones(&[1, 3]));

    // 第二次加载
    let loaded2 = Graph::load_model(path).unwrap();
    let pred2 = predict(&loaded2, &Tensor::ones(&[1, 3]));

    assert_vec_close(&pred1, &pred2, 1e-6, "两次加载预测应相同");

    // 清理
    std::fs::remove_file("test_otm_load_twice.otm").ok();
}

/// 覆盖已有文件（保存同一路径两次）
#[test]
fn test_save_model_overwrite() {
    let path = "test_otm_overwrite";

    // 第一次保存
    let graph1 = Graph::new_with_seed(42);
    let x1 = graph1.input(&Tensor::ones(&[1, 2])).unwrap();
    let fc1 = Linear::new(&graph1, 2, 1, true, "fc").unwrap();
    let out1 = fc1.forward(&x1);
    graph1.save_model(path, &[&out1]).unwrap();

    // 第二次保存（不同 seed → 不同权重）
    let graph2 = Graph::new_with_seed(99);
    let x2 = graph2.input(&Tensor::ones(&[1, 2])).unwrap();
    let fc2 = Linear::new(&graph2, 2, 1, true, "fc").unwrap();
    let out2 = fc2.forward(&x2);
    graph2.save_model(path, &[&out2]).unwrap();

    // 加载应得到第二次保存的结果
    graph2.forward(&out2).unwrap();
    let expected = out2.value().unwrap().unwrap().to_vec();

    let loaded = Graph::load_model(path).unwrap();
    let loaded_pred = predict(&loaded, &Tensor::ones(&[1, 2]));
    assert_vec_close(&expected, &loaded_pred, 1e-5, "覆盖后预测");

    // 清理
    std::fs::remove_file("test_otm_overwrite.otm").ok();
}

/// 带多个激活函数的模型往返（验证各种节点类型序列化）
#[test]
fn test_save_load_various_activations() {
    let path = "test_otm_activations";

    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::new(&[0.5, -0.3, 0.8, -0.1], &[1, 4])).unwrap();

    let fc1 = Linear::new(&graph, 4, 4, true, "fc1").unwrap();
    let h = fc1.forward(&x).tanh(); // Tanh 激活

    let fc2 = Linear::new(&graph, 4, 4, true, "fc2").unwrap();
    let h = fc2.forward(&h).sigmoid(); // Sigmoid 激活

    let fc3 = Linear::new(&graph, 4, 2, true, "fc3").unwrap();
    let out = fc3.forward(&h);

    graph.forward(&out).unwrap();
    let original_pred = out.value().unwrap().unwrap().to_vec();

    graph.save_model(path, &[&out]).unwrap();
    let loaded = Graph::load_model(path).unwrap();
    let loaded_pred = predict(&loaded, &Tensor::new(&[0.5, -0.3, 0.8, -0.1], &[1, 4]));
    assert_vec_close(&original_pred, &loaded_pred, 1e-5, "多激活函数预测");

    // 清理
    std::fs::remove_file("test_otm_activations.otm").ok();
}

/// 模型参数数量验证
#[test]
fn test_save_load_parameter_count() {
    let path = "test_otm_param_count";

    let graph = Graph::new_with_seed(42);
    let x = graph.input(&Tensor::ones(&[1, 4])).unwrap();
    let fc1 = Linear::new(&graph, 4, 8, true, "fc1").unwrap(); // W:4*8 + b:8 = 40
    let fc2 = Linear::new(&graph, 8, 2, true, "fc2").unwrap(); // W:8*2 + b:2 = 18
    let out = fc2.forward(&fc1.forward(&x).relu());

    let original_count = graph.parameter_count();
    graph.save_model(path, &[&out]).unwrap();

    let loaded = Graph::load_model(path).unwrap();
    let loaded_count = loaded.graph.parameter_count();

    assert_eq!(original_count, loaded_count, "参数数量应一致");

    // 清理
    std::fs::remove_file("test_otm_param_count.otm").ok();
}
