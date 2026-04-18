/*
 * @Author       : 老董
 * @Date         : 2025-12-27
 * @Description  : Graph 参数保存/加载测试 + 可视化测试
 *
 * 覆盖参数保存/加载与可视化；与当前 API 的对应关系：
 * - describe() / summary_string() / summary_markdown() / save_summary() 已移除，相关测试已删除
 * - save_weights() 仅保存 .bin（不生成 JSON 描述文件）
 * - 可视化在 Var 上（to_dot / save_visualization），不再支持自定义图片格式
 * - 测试使用 Graph 与 inner_rc() 等底层接口
 */

use crate::nn::nodes::NodeInner;
use crate::nn::{Graph, GraphInner, ImageFormat, Var};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::cell::RefCell;
use std::fs;
use std::rc::Rc;

/// 辅助函数：创建参数节点并注册到图中
fn make_param(gi: &Rc<RefCell<GraphInner>>, shape: &[usize], name: &str) -> Rc<NodeInner> {
    let node = gi
        .borrow_mut()
        .create_parameter_node(shape, Some(name))
        .unwrap();
    gi.borrow_mut()
        .register_parameter(name.to_string(), Rc::downgrade(&node))
        .unwrap();
    node
}

// ========== save_params / load_params 测试 ==========

/// 测试基本的参数保存和加载
#[test]
fn test_save_load_params_basic() {
    let temp_file = "test_save_load_params_basic.bin";

    // 1. 创建图并设置参数值
    let graph = Graph::new();
    let gi = graph.inner_rc();

    let w1 = make_param(&gi, &[3, 4], "w1");
    let b1 = make_param(&gi, &[1, 4], "b1");

    // 设置特定值
    let w1_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let b1_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    w1.set_value(Some(&Tensor::new(&w1_data, &[3, 4])))
        .expect("设置 w1 值失败");
    b1.set_value(Some(&Tensor::new(&b1_data, &[1, 4])))
        .expect("设置 b1 值失败");

    // 2. 保存参数
    gi.borrow().save_params(temp_file).expect("保存参数失败");

    // 3. 创建新图并加载参数
    let graph2 = Graph::new();
    let gi2 = graph2.inner_rc();

    let w1_new = make_param(&gi2, &[3, 4], "w1");
    let b1_new = make_param(&gi2, &[1, 4], "b1");

    gi2.borrow_mut()
        .load_params(temp_file)
        .expect("加载参数失败");

    // 4. 验证参数值
    let w1_loaded = w1_new.value().unwrap();
    let b1_loaded = b1_new.value().unwrap();

    assert_eq!(w1_loaded.shape(), &[3, 4]);
    assert_eq!(b1_loaded.shape(), &[1, 4]);

    for (i, &val) in w1_loaded.data_as_slice().iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32 * 0.1, epsilon = 1e-6);
    }
    assert_abs_diff_eq!(&b1_loaded, &Tensor::new(&b1_data, &[1, 4]), epsilon = 1e-6);

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试部分参数加载（迁移学习场景）
#[test]
fn test_save_load_params_partial() {
    let temp_file = "test_save_load_params_partial.bin";

    // 1. 创建包含 3 个参数的图
    let graph1 = Graph::new();
    let gi1 = graph1.inner_rc();

    let _w1 = make_param(&gi1, &[2, 3], "w1");
    let _w2 = make_param(&gi1, &[3, 4], "w2");
    let _w3 = make_param(&gi1, &[4, 5], "w3");

    gi1.borrow().save_params(temp_file).expect("保存参数失败");

    // 2. 创建只有 2 个参数的图（缺少 w2）
    let graph2 = Graph::new();
    let gi2 = graph2.inner_rc();

    let w1_new = make_param(&gi2, &[2, 3], "w1");
    let w3_new = make_param(&gi2, &[4, 5], "w3");

    // 加载应该成功，w2 会被忽略
    gi2.borrow_mut()
        .load_params(temp_file)
        .expect("加载参数失败");

    // 验证 w1 和 w3 被加载
    assert!(w1_new.value().is_some());
    assert!(w3_new.value().is_some());

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试无效文件格式检测
#[test]
fn test_load_params_invalid_magic() {
    let temp_file = "test_load_params_invalid_magic.bin";

    // 写入无效数据
    fs::write(temp_file, b"INVALID_DATA").expect("写入测试文件失败");

    let graph = Graph::new();
    let gi = graph.inner_rc();
    let _w1 = make_param(&gi, &[2, 3], "w1");

    let result = gi.borrow_mut().load_params(temp_file);
    assert!(result.is_err());
    // 使用 Debug 格式检查错误信息
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(err_msg.contains("无效的参数文件"), "错误信息: {}", err_msg);

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试文件不存在的错误处理
#[test]
fn test_load_params_file_not_found() {
    let graph = Graph::new();
    let gi = graph.inner_rc();
    let _w1 = make_param(&gi, &[2, 3], "w1");

    let result = gi.borrow_mut().load_params("nonexistent_file.bin");
    assert!(result.is_err());
    // 使用 Debug 格式检查错误信息
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("无法打开参数文件"),
        "错误信息: {}",
        err_msg
    );
}

/// 测试空图的保存和加载
#[test]
fn test_save_load_params_empty_graph() {
    let temp_file = "test_save_load_params_empty.bin";

    // 创建没有参数节点的图
    let graph = Graph::new();
    let gi = graph.inner_rc();
    gi.borrow()
        .save_params(temp_file)
        .expect("保存空图参数失败");

    // 加载到另一个空图
    let graph2 = Graph::new();
    let gi2 = graph2.inner_rc();
    gi2.borrow_mut()
        .load_params(temp_file)
        .expect("加载空图参数失败");

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试中文名称参数的保存加载（验证 UTF-8 编码）
#[test]
fn test_save_load_params_utf8_names() {
    let temp_file = "test_save_load_params_utf8.bin";

    let graph = Graph::new();
    let gi = graph.inner_rc();

    let w1 = make_param(&gi, &[2, 3], "权重_层1");
    let b1 = make_param(&gi, &[1, 3], "偏置_层1");

    // 设置值
    let w1_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b1_data: Vec<f32> = vec![0.5, 1.5, 2.5];
    w1.set_value(Some(&Tensor::new(&w1_data, &[2, 3]))).unwrap();
    b1.set_value(Some(&Tensor::new(&b1_data, &[1, 3]))).unwrap();

    gi.borrow().save_params(temp_file).expect("保存参数失败");

    // 新图加载
    let graph2 = Graph::new();
    let gi2 = graph2.inner_rc();

    let w1_new = make_param(&gi2, &[2, 3], "权重_层1");
    let b1_new = make_param(&gi2, &[1, 3], "偏置_层1");

    gi2.borrow_mut()
        .load_params(temp_file)
        .expect("加载参数失败");

    // 验证
    let w1_loaded = w1_new.value().unwrap();
    let b1_loaded = b1_new.value().unwrap();
    assert_abs_diff_eq!(&w1_loaded, &Tensor::new(&w1_data, &[2, 3]), epsilon = 1e-6);
    assert_abs_diff_eq!(&b1_loaded, &Tensor::new(&b1_data, &[1, 3]), epsilon = 1e-6);

    // 清理
    fs::remove_file(temp_file).ok();
}

// ========== save_weights / load_weights 测试 ==========

/// 测试权重保存和加载
#[test]
fn test_save_load_weights() {
    let temp_base = "test_save_load_weights";

    // 1. 创建并训练一个简单模型
    let graph = Graph::new();
    let gi = graph.inner_rc();

    let _x = gi
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("x"))
        .unwrap();
    let w = make_param(&gi, &[4, 2], "w");
    let b = make_param(&gi, &[1, 2], "b");

    // 设置特定的参数值
    let w_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = vec![1.0, 2.0];
    w.set_value(Some(&Tensor::new(&w_data, &[4, 2]))).unwrap();
    b.set_value(Some(&Tensor::new(&b_data, &[1, 2]))).unwrap();

    // 2. 保存权重
    gi.borrow().save_weights(temp_base).expect("保存权重失败");

    // 验证 .bin 文件存在
    assert!(std::path::Path::new("test_save_load_weights.bin").exists());

    // 3. 创建相同结构的新图并加载模型
    let graph2 = Graph::new();
    let gi2 = graph2.inner_rc();

    let _x2 = gi2
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("x"))
        .unwrap();
    let w2 = make_param(&gi2, &[4, 2], "w");
    let b2 = make_param(&gi2, &[1, 2], "b");

    gi2.borrow_mut()
        .load_weights(temp_base)
        .expect("加载权重失败");

    // 4. 验证参数值
    let w_loaded = w2.value().unwrap();
    let b_loaded = b2.value().unwrap();
    assert_abs_diff_eq!(&w_loaded, &Tensor::new(&w_data, &[4, 2]), epsilon = 1e-6);
    assert_abs_diff_eq!(&b_loaded, &Tensor::new(&b_data, &[1, 2]), epsilon = 1e-6);

    // 清理
    fs::remove_file("test_save_load_weights.bin").ok();
}

/// 测试 .bin 文件内容有效
#[test]
fn test_save_weights_bin_valid() {
    let temp_base = "test_save_weights_bin_valid";

    let graph = Graph::new();
    let gi = graph.inner_rc();

    let _input = gi
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let _weight = make_param(&gi, &[3, 4], "weight");

    gi.borrow().save_weights(temp_base).expect("保存权重失败");

    // 验证 .bin 文件存在且有内容
    let bin_path = "test_save_weights_bin_valid.bin";
    let content = fs::read(bin_path).expect("读取 bin 文件失败");

    // 验证魔数 "OTPR"
    assert_eq!(&content[0..4], b"OTPR", "应包含正确的魔数");
    // 验证版本号 (u32 = 1)
    assert_eq!(
        u32::from_le_bytes(content[4..8].try_into().unwrap()),
        1,
        "版本号应为 1"
    );
    // 验证参数数量 (u32 = 1, 只有 weight)
    assert_eq!(
        u32::from_le_bytes(content[8..12].try_into().unwrap()),
        1,
        "应包含 1 个参数"
    );
    // 文件应大于 header（12 字节）
    assert!(content.len() > 12, "文件应包含参数数据");

    // 清理
    fs::remove_file(bin_path).ok();
}

// ========== to_dot / 可视化测试 ==========

/// 测试 DOT 输出基本格式
#[test]
fn test_to_dot_basic() {
    let graph = Graph::new();
    let gi = graph.inner_rc();

    // 构建简单的网络：input -> matmul(w) -> loss
    let x = gi
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("input"))
        .unwrap();
    let w = gi
        .borrow_mut()
        .create_parameter_node(&[4, 2], Some("weight"))
        .unwrap();
    let z = gi
        .borrow_mut()
        .create_mat_mul_node(vec![Rc::clone(&x), Rc::clone(&w)], Some("output"))
        .unwrap();
    let target = gi
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("target"))
        .unwrap();
    let loss = gi
        .borrow_mut()
        .create_mse_mean_node(Rc::clone(&z), Rc::clone(&target), Some("loss"))
        .unwrap();

    let loss_var = Var::new_with_rc_graph(loss, &gi);
    let dot = loss_var.to_dot();

    // 验证 DOT 格式元素
    assert!(dot.contains("digraph ComputeGraph"));
    assert!(dot.contains("rankdir=TB"));
    assert!(dot.contains("input"));
    assert!(dot.contains("weight"));
    assert!(dot.contains("output"));
    assert!(dot.contains("loss"));

    // 验证边存在
    assert!(dot.contains("->"));

    // 验证节点样式
    assert!(dot.contains("ellipse")); // Input
    assert!(dot.contains("box")); // Parameter / 运算节点
    assert!(dot.contains("fillcolor"));

    // 打印 DOT 输出（用于手动检查）
    println!("\n{}", dot);
}

/// 测试 save_visualization 基本功能
#[test]
fn test_save_visualization_basic() {
    let base_path = "test_save_visualization_basic";

    let graph = Graph::new();
    let gi = graph.inner_rc();

    let x = gi
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let y = gi
        .borrow_mut()
        .create_sigmoid_node(Rc::clone(&x), Some("y"))
        .unwrap();

    let y_var = Var::new_with_rc_graph(y, &gi);
    let result = y_var
        .save_visualization(base_path)
        .expect("save_visualization 失败");

    // 验证 .dot 文件始终生成
    assert!(result.dot_path.exists(), "DOT 文件应该存在");
    let content = fs::read_to_string(&result.dot_path).expect("读取 DOT 文件失败");
    assert!(content.contains("digraph ComputeGraph"));
    assert!(content.contains("x"));
    assert!(content.contains("y"));

    // 如果 Graphviz 可用，验证图像文件
    if result.graphviz_available {
        if let Some(ref img_path) = result.image_path {
            assert!(img_path.exists(), "图像文件应该存在");
            assert!(
                img_path.extension().map(|e| e == "png").unwrap_or(false),
                "默认格式应为 PNG"
            );
            // 清理图像文件
            fs::remove_file(img_path).ok();
        }
    } else {
        assert!(result.graphviz_hint.is_some(), "应提供安装提示");
        println!("Graphviz 未安装，提示: {}", result.graphviz_hint.unwrap());
    }

    // 清理 DOT 文件
    fs::remove_file(&result.dot_path).ok();
}

/// 测试路径包含后缀时应报错
#[test]
fn test_save_visualization_rejects_extension() {
    let graph = Graph::new();
    let gi = graph.inner_rc();

    let x = gi
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("x"))
        .unwrap();
    let y = gi
        .borrow_mut()
        .create_sigmoid_node(Rc::clone(&x), Some("y"))
        .unwrap();

    let y_var = Var::new_with_rc_graph(y, &gi);

    // 测试 .dot 后缀
    let result = y_var.save_visualization("output/model.dot");
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("不应包含文件后缀"),
        "错误信息应提示不含后缀: {}",
        err_msg
    );

    // 测试 .png 后缀
    let result = y_var.save_visualization("output/model.png");
    assert!(result.is_err());

    // 测试未知后缀
    let result = y_var.save_visualization("output/model.xyz");
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("不应包含文件后缀"),
        "错误信息应提示不含后缀: {}",
        err_msg
    );
}

/// 测试 ImageFormat 枚举
#[test]
fn test_image_format() {
    // 测试扩展名
    assert_eq!(ImageFormat::Png.extension(), "png");
    assert_eq!(ImageFormat::Svg.extension(), "svg");
    assert_eq!(ImageFormat::Pdf.extension(), "pdf");

    // 测试从扩展名解析
    assert_eq!(ImageFormat::from_extension("png"), Some(ImageFormat::Png));
    assert_eq!(ImageFormat::from_extension("PNG"), Some(ImageFormat::Png));
    assert_eq!(ImageFormat::from_extension("svg"), Some(ImageFormat::Svg));
    assert_eq!(ImageFormat::from_extension("pdf"), Some(ImageFormat::Pdf));
    assert_eq!(ImageFormat::from_extension("unknown"), None);

    // 测试默认值
    assert_eq!(ImageFormat::default(), ImageFormat::Png);
}

/// 测试各节点类型的样式
#[test]
fn test_to_dot_node_styles() {
    let graph = Graph::new();
    let gi = graph.inner_rc();

    // 各种类型节点（所有节点连通，确保 Var::to_dot 能遍历到）
    let x = gi
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("input"))
        .unwrap();
    let w = gi
        .borrow_mut()
        .create_parameter_node(&[4, 2], Some("param"))
        .unwrap();
    let z = gi
        .borrow_mut()
        .create_mat_mul_node(vec![Rc::clone(&x), Rc::clone(&w)], Some("matmul"))
        .unwrap();
    let sigmoid = gi
        .borrow_mut()
        .create_sigmoid_node(Rc::clone(&z), Some("sigmoid"))
        .unwrap();
    let target = gi
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("target"))
        .unwrap();
    let loss = gi
        .borrow_mut()
        .create_mse_mean_node(Rc::clone(&sigmoid), Rc::clone(&target), Some("loss"))
        .unwrap();

    let loss_var = Var::new_with_rc_graph(loss, &gi);
    let dot = loss_var.to_dot();

    // 输入节点应该是椭圆形
    assert!(dot.contains("ellipse") && dot.contains("#E3F2FD"));
    // 参数节点应该是矩形绿色
    assert!(dot.contains("#E8F5E9"));
    // 激活函数应该是菱形橙色
    assert!(dot.contains("diamond") && dot.contains("#FFF3E0"));
    // 损失节点应该是红色
    assert!(dot.contains("#FFEBEE"));
}

/// 测试可视化中动态形状的显示
#[test]
fn test_dynamic_shape_in_visualization() {
    let graph = Graph::new();
    let gi = graph.inner_rc();

    // 创建支持动态 batch 的节点
    let x = gi
        .borrow_mut()
        .create_basic_input_node(&[8, 32], Some("x"))
        .unwrap();
    let w = gi
        .borrow_mut()
        .create_parameter_node(&[32, 16], Some("w"))
        .unwrap();
    let y = gi
        .borrow_mut()
        .create_mat_mul_node(vec![Rc::clone(&x), Rc::clone(&w)], Some("y"))
        .unwrap();

    let y_var = Var::new_with_rc_graph(y, &gi);
    let dot = y_var.to_dot();

    // Input 和 MatMul 应该显示动态 batch [?, ...]
    assert!(
        dot.contains("[?, 32]"),
        "Input 节点应显示动态 batch: {}",
        dot
    );
    assert!(
        dot.contains("[?, 16]"),
        "MatMul 节点应显示动态 batch: {}",
        dot
    );
    // Parameter 应该显示固定形状
    assert!(
        dot.contains("[32, 16]"),
        "Parameter 节点应显示固定形状: {}",
        dot
    );
}
