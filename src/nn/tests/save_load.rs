/*
 * @Author       : 老董
 * @Date         : 2025-12-27
 * @Description  : Graph 参数保存/加载测试 + 图描述测试
 */

use crate::nn::{GraphInner, ImageFormat, NodeTypeDescriptor};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;
use std::fs;

/// 测试基本的参数保存和加载
#[test]
fn test_save_load_params_basic() {
    let temp_file = "test_save_load_params_basic.bin";

    // 1. 创建图并设置参数值
    let mut graph = GraphInner::new();
    let w1 = graph
        .new_parameter_node(&[3, 4], Some("w1"))
        .expect("创建 w1 失败");
    let b1 = graph
        .new_parameter_node(&[1, 4], Some("b1"))
        .expect("创建 b1 失败");

    // 设置特定值
    let w1_data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    let b1_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    graph
        .set_node_value(w1, Some(&Tensor::new(&w1_data, &[3, 4])))
        .expect("设置 w1 值失败");
    graph
        .set_node_value(b1, Some(&Tensor::new(&b1_data, &[1, 4])))
        .expect("设置 b1 值失败");

    // 2. 保存参数
    graph.save_params(temp_file).expect("保存参数失败");

    // 3. 创建新图并加载参数
    let mut graph2 = GraphInner::new();
    let w1_new = graph2
        .new_parameter_node(&[3, 4], Some("w1"))
        .expect("创建 w1 失败");
    let b1_new = graph2
        .new_parameter_node(&[1, 4], Some("b1"))
        .expect("创建 b1 失败");

    graph2.load_params(temp_file).expect("加载参数失败");

    // 4. 验证参数值
    let w1_loaded = graph2.get_node_value(w1_new).unwrap().unwrap();
    let b1_loaded = graph2.get_node_value(b1_new).unwrap().unwrap();

    assert_eq!(w1_loaded.shape(), &[3, 4]);
    assert_eq!(b1_loaded.shape(), &[1, 4]);

    for (i, &val) in w1_loaded.data_as_slice().iter().enumerate() {
        assert_abs_diff_eq!(val, i as f32 * 0.1, epsilon = 1e-6);
    }
    assert_abs_diff_eq!(b1_loaded, &Tensor::new(&b1_data, &[1, 4]), epsilon = 1e-6);

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试部分参数加载（迁移学习场景）
#[test]
fn test_save_load_params_partial() {
    let temp_file = "test_save_load_params_partial.bin";

    // 1. 创建包含 3 个参数的图
    let mut graph1 = GraphInner::new();
    let _w1 = graph1
        .new_parameter_node(&[2, 3], Some("w1"))
        .expect("创建 w1 失败");
    let _w2 = graph1
        .new_parameter_node(&[3, 4], Some("w2"))
        .expect("创建 w2 失败");
    let _w3 = graph1
        .new_parameter_node(&[4, 5], Some("w3"))
        .expect("创建 w3 失败");

    graph1.save_params(temp_file).expect("保存参数失败");

    // 2. 创建只有 2 个参数的图（缺少 w2）
    let mut graph2 = GraphInner::new();
    let w1_new = graph2
        .new_parameter_node(&[2, 3], Some("w1"))
        .expect("创建 w1 失败");
    let w3_new = graph2
        .new_parameter_node(&[4, 5], Some("w3"))
        .expect("创建 w3 失败");

    // 加载应该成功，w2 会被忽略
    graph2.load_params(temp_file).expect("加载参数失败");

    // 验证 w1 和 w3 被加载
    assert!(graph2.get_node_value(w1_new).unwrap().is_some());
    assert!(graph2.get_node_value(w3_new).unwrap().is_some());

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试无效文件格式检测
#[test]
fn test_load_params_invalid_magic() {
    let temp_file = "test_load_params_invalid_magic.bin";

    // 写入无效数据
    fs::write(temp_file, b"INVALID_DATA").expect("写入测试文件失败");

    let mut graph = GraphInner::new();
    let _w1 = graph
        .new_parameter_node(&[2, 3], Some("w1"))
        .expect("创建 w1 失败");

    let result = graph.load_params(temp_file);
    assert!(result.is_err());
    // 使用 Debug 格式检查错误信息
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("无效的参数文件"),
        "错误信息: {}",
        err_msg
    );

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试文件不存在的错误处理
#[test]
fn test_load_params_file_not_found() {
    let mut graph = GraphInner::new();
    let _w1 = graph
        .new_parameter_node(&[2, 3], Some("w1"))
        .expect("创建 w1 失败");

    let result = graph.load_params("nonexistent_file.bin");
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
    let graph = GraphInner::new();
    graph.save_params(temp_file).expect("保存空图参数失败");

    // 加载到另一个空图
    let mut graph2 = GraphInner::new();
    graph2.load_params(temp_file).expect("加载空图参数失败");

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试中文名称参数的保存加载（验证 UTF-8 编码）
#[test]
fn test_save_load_params_utf8_names() {
    let temp_file = "test_save_load_params_utf8.bin";

    let mut graph = GraphInner::new();
    let w1 = graph
        .new_parameter_node(&[2, 3], Some("权重_层1"))
        .expect("创建参数失败");
    let b1 = graph
        .new_parameter_node(&[1, 3], Some("偏置_层1"))
        .expect("创建参数失败");

    // 设置值
    let w1_data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let b1_data: Vec<f32> = vec![0.5, 1.5, 2.5];
    graph
        .set_node_value(w1, Some(&Tensor::new(&w1_data, &[2, 3])))
        .unwrap();
    graph
        .set_node_value(b1, Some(&Tensor::new(&b1_data, &[1, 3])))
        .unwrap();

    graph.save_params(temp_file).expect("保存参数失败");

    // 新图加载
    let mut graph2 = GraphInner::new();
    let w1_new = graph2
        .new_parameter_node(&[2, 3], Some("权重_层1"))
        .expect("创建参数失败");
    let b1_new = graph2
        .new_parameter_node(&[1, 3], Some("偏置_层1"))
        .expect("创建参数失败");

    graph2.load_params(temp_file).expect("加载参数失败");

    // 验证
    let w1_loaded = graph2.get_node_value(w1_new).unwrap().unwrap();
    let b1_loaded = graph2.get_node_value(b1_new).unwrap().unwrap();
    assert_abs_diff_eq!(w1_loaded, &Tensor::new(&w1_data, &[2, 3]), epsilon = 1e-6);
    assert_abs_diff_eq!(b1_loaded, &Tensor::new(&b1_data, &[1, 3]), epsilon = 1e-6);

    // 清理
    fs::remove_file(temp_file).ok();
}

// ========== GraphDescriptor 测试 ==========

/// 测试基本的图描述
#[test]
fn test_describe_basic() {
    let mut graph = GraphInner::new();

    // 构建简单的网络：input -> matmul(w) -> sigmoid -> loss
    let x = graph.new_basic_input_node(&[1, 4], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[4, 2], Some("w")).unwrap();
    let z = graph.new_mat_mul_node(x, w, Some("z")).unwrap();
    let _a = graph.new_sigmoid_node(z, Some("a")).unwrap();

    let desc = graph.describe();

    // 验证基本信息
    assert_eq!(desc.nodes.len(), 4);
    assert!(!desc.version.is_empty());

    // 验证节点类型
    let x_desc = desc.nodes.iter().find(|n| n.name == "x").unwrap();
    assert!(matches!(x_desc.node_type, NodeTypeDescriptor::BasicInput));
    assert_eq!(x_desc.output_shape, vec![1, 4]);
    assert!(x_desc.parents.is_empty());

    let w_desc = desc.nodes.iter().find(|n| n.name == "w").unwrap();
    assert!(matches!(w_desc.node_type, NodeTypeDescriptor::Parameter));
    assert_eq!(w_desc.param_count, Some(8)); // 4 * 2

    let z_desc = desc.nodes.iter().find(|n| n.name == "z").unwrap();
    assert!(matches!(z_desc.node_type, NodeTypeDescriptor::MatMul));
    assert_eq!(z_desc.parents.len(), 2);

    let a_desc = desc.nodes.iter().find(|n| n.name == "a").unwrap();
    assert!(matches!(a_desc.node_type, NodeTypeDescriptor::Sigmoid));
    assert_eq!(a_desc.parents.len(), 1);
}

/// 测试 JSON 序列化/反序列化
#[test]
fn test_describe_json_roundtrip() {
    let mut graph = GraphInner::new();

    let x = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let w = graph.new_parameter_node(&[3, 4], Some("weight")).unwrap();
    let _ = graph.new_mat_mul_node(x, w, Some("output")).unwrap();

    let desc = graph.describe();
    let json = desc.to_json().expect("序列化失败");

    // 反序列化
    let desc2 = crate::nn::GraphDescriptor::from_json(&json).expect("反序列化失败");

    assert_eq!(desc.nodes.len(), desc2.nodes.len());
    assert_eq!(desc.version, desc2.version);
}

/// 测试总参数量计算
#[test]
fn test_describe_total_params() {
    let mut graph = GraphInner::new();

    let _ = graph.new_basic_input_node(&[1, 784], Some("x")).unwrap();
    let _ = graph.new_parameter_node(&[784, 128], Some("w1")).unwrap(); // 100352
    let _ = graph.new_parameter_node(&[1, 128], Some("b1")).unwrap(); // 128
    let _ = graph.new_parameter_node(&[128, 10], Some("w2")).unwrap(); // 1280
    let _ = graph.new_parameter_node(&[1, 10], Some("b2")).unwrap(); // 10

    let desc = graph.describe();
    assert_eq!(desc.total_params(), 100352 + 128 + 1280 + 10);
}

// ========== save_model / load_model 测试 ==========

/// 测试完整模型保存和加载
#[test]
fn test_save_load_model() {
    let temp_base = "test_save_load_model";

    // 1. 创建并训练一个简单模型
    let mut graph = GraphInner::new();
    let _x = graph.new_basic_input_node(&[1, 4], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[4, 2], Some("w")).unwrap();
    let b = graph.new_parameter_node(&[1, 2], Some("b")).unwrap();

    // 设置特定的参数值
    let w_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
    let b_data: Vec<f32> = vec![1.0, 2.0];
    graph
        .set_node_value(w, Some(&Tensor::new(&w_data, &[4, 2])))
        .unwrap();
    graph
        .set_node_value(b, Some(&Tensor::new(&b_data, &[1, 2])))
        .unwrap();

    // 2. 保存模型
    graph.save_model(temp_base).expect("保存模型失败");

    // 验证文件存在
    assert!(std::path::Path::new("test_save_load_model.json").exists());
    assert!(std::path::Path::new("test_save_load_model.bin").exists());

    // 3. 创建相同结构的新图并加载模型
    let mut graph2 = GraphInner::new();
    let _x2 = graph2.new_basic_input_node(&[1, 4], Some("x")).unwrap();
    let w2 = graph2.new_parameter_node(&[4, 2], Some("w")).unwrap();
    let b2 = graph2.new_parameter_node(&[1, 2], Some("b")).unwrap();

    graph2.load_model(temp_base).expect("加载模型失败");

    // 4. 验证参数值
    let w_loaded = graph2.get_node_value(w2).unwrap().unwrap();
    let b_loaded = graph2.get_node_value(b2).unwrap().unwrap();
    assert_abs_diff_eq!(w_loaded, &Tensor::new(&w_data, &[4, 2]), epsilon = 1e-6);
    assert_abs_diff_eq!(b_loaded, &Tensor::new(&b_data, &[1, 2]), epsilon = 1e-6);

    // 清理
    fs::remove_file("test_save_load_model.json").ok();
    fs::remove_file("test_save_load_model.bin").ok();
}

/// 测试 JSON 文件内容可读
#[test]
fn test_save_model_json_readable() {
    let temp_base = "test_save_model_json_readable";

    let mut graph = GraphInner::new();
    let _ = graph.new_basic_input_node(&[2, 3], Some("input")).unwrap();
    let _ = graph.new_parameter_node(&[3, 4], Some("weight")).unwrap();

    graph.save_model(temp_base).expect("保存模型失败");

    // 读取并验证 JSON 内容
    let json = fs::read_to_string("test_save_model_json_readable.json").expect("读取 JSON 失败");
    assert!(json.contains("\"version\""));
    assert!(json.contains("\"nodes\""));
    assert!(json.contains("\"input\""));
    assert!(json.contains("\"weight\""));
    assert!(json.contains("\"params_file\""));

    // 清理
    fs::remove_file("test_save_model_json_readable.json").ok();
    fs::remove_file("test_save_model_json_readable.bin").ok();
}

// ========== summary 测试 ==========

/// 测试 summary 输出基本格式
#[test]
fn test_summary_basic() {
    let mut graph = GraphInner::new();

    let x = graph.new_basic_input_node(&[1, 784], Some("x")).unwrap();
    let w1 = graph.new_parameter_node(&[784, 128], Some("w1")).unwrap();
    let b1 = graph.new_parameter_node(&[1, 128], Some("b1")).unwrap();
    let z1 = graph.new_mat_mul_node(x, w1, Some("z1")).unwrap();
    let a1 = graph.new_add_node(&[z1, b1], Some("a1")).unwrap();
    let _h1 = graph.new_sigmoid_node(a1, Some("h1")).unwrap();

    let summary = graph.summary_string();

    // 验证表格元素存在
    assert!(summary.contains("节点名称"));
    assert!(summary.contains("类型"));
    assert!(summary.contains("输出形状"));
    assert!(summary.contains("参数量"));
    assert!(summary.contains("父节点"));

    // 验证节点信息
    assert!(summary.contains("x"));
    assert!(summary.contains("Input"));
    assert!(summary.contains("w1"));
    assert!(summary.contains("Parameter"));
    assert!(summary.contains("MatMul"));
    assert!(summary.contains("Sigmoid"));

    // 验证参数统计
    assert!(summary.contains("总参数量"));
    assert!(summary.contains("可训练参数"));

    // 打印输出（用于手动检查格式）
    println!("\n{}", summary);
}

/// 测试参数量格式化（千分位分隔）
#[test]
fn test_summary_param_formatting() {
    let mut graph = GraphInner::new();

    let _ = graph
        .new_parameter_node(&[1000, 1000], Some("big_param"))
        .unwrap(); // 1,000,000

    let summary = graph.summary_string();

    // 验证千分位分隔
    assert!(summary.contains("1,000,000"), "参数量应使用千分位分隔");
}

/// 测试空图的 summary
#[test]
fn test_summary_empty_graph() {
    let graph = GraphInner::new();
    let summary = graph.summary_string();

    // 空图也应该能输出表格结构
    assert!(summary.contains("节点名称"));
    assert!(summary.contains("总参数量: 0"));
}

/// 测试 summary 保存到文本文件
#[test]
fn test_save_summary_txt() {
    let temp_file = "test_save_summary.txt";

    let mut graph = GraphInner::new();
    let x = graph.new_basic_input_node(&[1, 4], Some("x")).unwrap();
    let _ = graph.new_sigmoid_node(x, Some("y")).unwrap();

    graph.save_summary(temp_file).expect("保存摘要失败");

    // 验证文件内容（Unicode 表格格式）
    let content = fs::read_to_string(temp_file).expect("读取摘要文件失败");
    assert!(content.contains("节点名称"));
    assert!(content.contains("x"));
    assert!(content.contains("y"));
    assert!(content.contains("总参数量"));
    assert!(content.contains("┌")); // Unicode 表格边框

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试 summary 保存到 Markdown 文件
#[test]
fn test_save_summary_markdown() {
    let temp_file = "test_save_summary.md";

    let mut graph = GraphInner::new();
    let x = graph.new_basic_input_node(&[1, 4], Some("x")).unwrap();
    let _ = graph.new_sigmoid_node(x, Some("y")).unwrap();

    graph.save_summary(temp_file).expect("保存摘要失败");

    // 验证 Markdown 格式
    let content = fs::read_to_string(temp_file).expect("读取摘要文件失败");
    assert!(content.contains("# 模型摘要")); // Markdown 标题
    assert!(content.contains("|-------")); // Markdown 表格分隔符
    assert!(content.contains("| x |")); // Markdown 表格行
    assert!(content.contains("**总参数量**")); // Markdown 粗体

    // 清理
    fs::remove_file(temp_file).ok();
}

/// 测试 summary_markdown 方法
#[test]
fn test_summary_markdown_string() {
    let mut graph = GraphInner::new();
    let _ = graph.new_parameter_node(&[10, 20], Some("weight")).unwrap();

    let md = graph.summary_markdown();

    // 验证 Markdown 格式元素
    assert!(md.contains("# 模型摘要"));
    assert!(md.contains("| 节点名称 | 类型 |"));
    assert!(md.contains("| weight | Parameter |"));
    assert!(md.contains("| 200 |")); // 参数量 10*20=200
    assert!(md.contains("**总参数量**: 200"));
}

// ========== to_dot 测试 ==========

/// 测试 DOT 输出基本格式
#[test]
fn test_to_dot_basic() {
    let mut graph = GraphInner::new();

    let x = graph.new_basic_input_node(&[1, 4], Some("input")).unwrap();
    let w = graph.new_parameter_node(&[4, 2], Some("weight")).unwrap();
    let z = graph.new_mat_mul_node(x, w, Some("output")).unwrap();
    let target = graph.new_basic_input_node(&[1, 2], Some("target")).unwrap();
    let _loss = graph.new_mse_loss_node(z, target, Some("loss")).unwrap();

    let dot = graph.to_dot();

    // 验证 DOT 格式元素
    assert!(dot.contains("digraph Model"));
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

    let mut graph = GraphInner::new();
    let x = graph.new_basic_input_node(&[2, 3], Some("x")).unwrap();
    let _ = graph.new_sigmoid_node(x, Some("y")).unwrap();

    let result = graph
        .save_visualization(base_path, None)
        .expect("save_visualization 失败");

    // 验证 .dot 文件始终生成
    assert!(result.dot_path.exists(), "DOT 文件应该存在");
    let content = fs::read_to_string(&result.dot_path).expect("读取 DOT 文件失败");
    assert!(content.contains("digraph Model"));
    assert!(content.contains("x"));
    assert!(content.contains("y"));

    // 如果 Graphviz 可用，验证图像文件
    if result.graphviz_available {
        assert!(result.image_path.is_some(), "Graphviz 可用时应生成图像路径");
        let img_path = result.image_path.as_ref().unwrap();
        assert!(img_path.exists(), "图像文件应该存在");
        assert!(
            img_path.extension().map(|e| e == "png").unwrap_or(false),
            "默认格式应为 PNG"
        );
        // 清理图像文件
        fs::remove_file(img_path).ok();
    } else {
        assert!(result.graphviz_hint.is_some(), "应提供安装提示");
        println!("Graphviz 未安装，提示: {}", result.graphviz_hint.unwrap());
    }

    // 清理 DOT 文件
    fs::remove_file(&result.dot_path).ok();
}

/// 测试指定图像格式
#[test]
fn test_save_visualization_with_format() {
    let base_path = "test_save_visualization_svg";

    let mut graph = GraphInner::new();
    let x = graph.new_basic_input_node(&[2, 3], Some("x")).unwrap();
    let _ = graph.new_sigmoid_node(x, Some("y")).unwrap();

    let result = graph
        .save_visualization(base_path, Some(ImageFormat::Svg))
        .expect("save_visualization 失败");

    // 验证 .dot 文件
    assert!(result.dot_path.exists());

    // 如果 Graphviz 可用，验证 SVG 格式
    if result.graphviz_available {
        let img_path = result.image_path.as_ref().unwrap();
        assert!(
            img_path.extension().map(|e| e == "svg").unwrap_or(false),
            "应为 SVG 格式"
        );
        fs::remove_file(img_path).ok();
    }

    fs::remove_file(&result.dot_path).ok();
}

/// 测试路径包含后缀时应报错
#[test]
fn test_save_visualization_rejects_extension() {
    let mut graph = GraphInner::new();
    let x = graph.new_basic_input_node(&[2, 3], Some("x")).unwrap();
    let _ = graph.new_sigmoid_node(x, Some("y")).unwrap();

    // 测试 .dot 后缀
    let result = graph.save_visualization("output/model.dot", None);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("不含后缀"),
        "错误信息应提示不要加后缀: {}",
        err_msg
    );

    // 测试 .png 后缀
    let result = graph.save_visualization("output/model.png", None);
    assert!(result.is_err());

    // 测试未知后缀
    let result = graph.save_visualization("output/model.xyz", None);
    assert!(result.is_err());
    let err_msg = format!("{:?}", result.unwrap_err());
    assert!(
        err_msg.contains("未知后缀"),
        "错误信息应提示未知后缀: {}",
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
    let mut graph = GraphInner::new();

    // 各种类型节点
    let x = graph.new_basic_input_node(&[1, 4], Some("input")).unwrap();
    let _ = graph.new_parameter_node(&[4, 2], Some("param")).unwrap();
    let y = graph.new_sigmoid_node(x, Some("sigmoid")).unwrap();
    let target = graph.new_basic_input_node(&[1, 4], Some("target")).unwrap();
    let _ = graph.new_mse_loss_node(y, target, Some("loss")).unwrap();

    let dot = graph.to_dot();

    // 输入节点应该是椭圆形
    assert!(dot.contains("ellipse") && dot.contains("#E3F2FD"));
    // 参数节点应该是矩形绿色
    assert!(dot.contains("#E8F5E9"));
    // 激活函数应该是菱形橙色
    assert!(dot.contains("diamond") && dot.contains("#FFF3E0"));
    // 损失节点应该是红色
    assert!(dot.contains("#FFEBEE"));
}

/// 测试动态形状在描述符中的序列化
#[test]
fn test_dynamic_shape_in_descriptor() {
    use crate::nn::descriptor::NodeDescriptor;

    let mut graph = GraphInner::new();

    // 创建支持动态 batch 的节点
    let x = graph
        .new_basic_input_node(&[32, 128], Some("input"))
        .unwrap();
    let w = graph
        .new_parameter_node(&[128, 64], Some("weight"))
        .unwrap();
    let y = graph.new_mat_mul_node(x, w, Some("output")).unwrap();

    // 获取描述符
    let desc = graph.describe();

    // 查找各节点
    let input_node = desc.nodes.iter().find(|n| n.name == "input").unwrap();
    let weight_node = desc.nodes.iter().find(|n| n.name == "weight").unwrap();
    let output_node = desc.nodes.iter().find(|n| n.name == "output").unwrap();

    // Input 节点应该有动态形状（第一维是 None）
    assert!(input_node.dynamic_shape.is_some(), "Input 节点应有动态形状");
    let input_dyn = input_node.dynamic_shape.as_ref().unwrap();
    assert_eq!(input_dyn[0], None, "Input 的第一维应该是动态的");
    assert_eq!(input_dyn[1], Some(128), "Input 的第二维应该是固定的");

    // Parameter 节点不应该有动态形状
    assert!(
        weight_node.dynamic_shape.is_none(),
        "Parameter 节点不应有动态形状"
    );

    // MatMul 输出应该继承动态 batch
    assert!(
        output_node.dynamic_shape.is_some(),
        "MatMul 输出应有动态形状"
    );
    let output_dyn = output_node.dynamic_shape.as_ref().unwrap();
    assert_eq!(output_dyn[0], None, "MatMul 输出的第一维应该是动态的");
    assert_eq!(output_dyn[1], Some(64), "MatMul 输出的第二维应该是固定的");

    // 测试 display_shape
    assert_eq!(input_node.display_shape(), "[?, 128]");
    assert_eq!(weight_node.display_shape(), "[128, 64]"); // 固定形状
    assert_eq!(output_node.display_shape(), "[?, 64]");
}

/// 测试动态形状在 JSON 中的序列化/反序列化
#[test]
fn test_dynamic_shape_json_roundtrip() {
    use crate::nn::descriptor::GraphDescriptor;

    let mut graph = GraphInner::new();

    // 创建支持动态 batch 的图
    let x = graph.new_basic_input_node(&[16, 64], Some("x")).unwrap();
    let y = graph.new_tanh_node(x, Some("y")).unwrap();

    // 获取描述符并序列化为 JSON
    let desc = graph.describe();
    let json = desc.to_json().expect("JSON 序列化失败");

    // 反序列化
    let desc2 = GraphDescriptor::from_json(&json).expect("JSON 反序列化失败");

    // 验证动态形状信息被正确保留
    let x_node = desc2.nodes.iter().find(|n| n.name == "x").unwrap();
    let y_node = desc2.nodes.iter().find(|n| n.name == "y").unwrap();

    // x 节点应该有动态形状
    assert!(x_node.dynamic_shape.is_some());
    let x_dyn = x_node.dynamic_shape.as_ref().unwrap();
    assert_eq!(x_dyn, &vec![None, Some(64)]);

    // y 节点也应该有动态形状（继承自 x）
    assert!(y_node.dynamic_shape.is_some());
    let y_dyn = y_node.dynamic_shape.as_ref().unwrap();
    assert_eq!(y_dyn, &vec![None, Some(64)]);

    // 验证 display_shape
    assert_eq!(x_node.display_shape(), "[?, 64]");
    assert_eq!(y_node.display_shape(), "[?, 64]");
}

/// 测试可视化中动态形状的显示
#[test]
fn test_dynamic_shape_in_visualization() {
    let mut graph = GraphInner::new();

    // 创建支持动态 batch 的节点
    let x = graph.new_basic_input_node(&[8, 32], Some("x")).unwrap();
    let w = graph.new_parameter_node(&[32, 16], Some("w")).unwrap();
    let _ = graph.new_mat_mul_node(x, w, Some("y")).unwrap();

    // 生成 DOT 格式
    let dot = graph.to_dot();

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
