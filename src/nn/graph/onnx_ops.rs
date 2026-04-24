/*
 * @Author       : 老董
 * @Date         : 2026-04-18
 * @Description  : ONNX ↔ NodeTypeDescriptor 双向算子映射表
 *
 * 导入方向：ONNX OpType → NodeTypeDescriptor（可能需要属性提取）
 * 导出方向：NodeTypeDescriptor → ONNX OpType + 属性
 *
 * 设计原则：
 * - 不支持的算子必须明确返回 Err，不允许静默忽略
 * - Gemm 只支持标准形式 (alpha=1, beta=1, transB=1)，展开为 MatMul + Add
 * - 训练专用节点（loss/target）在导出方向被标记为不可导出
 */

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::graph::onnx_error::OnnxError;
use onnx_rs::ast::{Attribute, OpType};

// ==================== 导入方向：ONNX → NodeTypeDescriptor ====================

/// 将 ONNX 算子映射为一个或多个 NodeTypeDescriptor。
///
/// 返回 `Vec` 是因为 Gemm 需要展开为 MatMul + Add 两个节点。
/// 大多数算子返回恰好一个元素。
pub fn onnx_op_to_descriptors(
    op_type: &OpType,
    attrs: &[Attribute],
    node_name: &str,
) -> Result<Vec<NodeTypeDescriptor>, OnnxError> {
    match op_type {
        // ─── 直接映射的激活函数 ───
        OpType::Relu => Ok(vec![NodeTypeDescriptor::ReLU]),
        OpType::Sigmoid => Ok(vec![NodeTypeDescriptor::Sigmoid]),
        OpType::Tanh => Ok(vec![NodeTypeDescriptor::Tanh]),
        OpType::Softmax => Ok(vec![NodeTypeDescriptor::Softmax]),
        OpType::LogSoftmax => Ok(vec![NodeTypeDescriptor::LogSoftmax]),
        OpType::Gelu => Ok(vec![NodeTypeDescriptor::Gelu]),
        OpType::Selu => Ok(vec![NodeTypeDescriptor::Selu]),
        OpType::Mish => Ok(vec![NodeTypeDescriptor::Mish]),
        OpType::HardSwish => Ok(vec![NodeTypeDescriptor::HardSwish]),
        OpType::HardSigmoid => Ok(vec![NodeTypeDescriptor::HardSigmoid]),
        OpType::Softplus => Ok(vec![NodeTypeDescriptor::SoftPlus]),
        OpType::Elu => {
            let alpha = find_attr_float(attrs, "alpha").unwrap_or(1.0);
            Ok(vec![NodeTypeDescriptor::Elu { alpha }])
        }
        OpType::LeakyRelu => {
            let alpha = find_attr_float(attrs, "alpha").unwrap_or(0.01);
            Ok(vec![NodeTypeDescriptor::LeakyReLU { alpha }])
        }

        // ─── 直接映射的算术运算 ───
        OpType::Add => Ok(vec![NodeTypeDescriptor::Add]),
        OpType::Sub => Ok(vec![NodeTypeDescriptor::Subtract]),
        OpType::Mul => Ok(vec![NodeTypeDescriptor::Multiply]),
        OpType::Div => Ok(vec![NodeTypeDescriptor::Divide]),
        OpType::Neg => Ok(vec![NodeTypeDescriptor::Negate]),
        OpType::MatMul => Ok(vec![NodeTypeDescriptor::MatMul]),

        // ─── 直接映射的数学运算 ───
        OpType::Abs => Ok(vec![NodeTypeDescriptor::Abs]),
        OpType::Exp => Ok(vec![NodeTypeDescriptor::Exp]),
        OpType::Sqrt => Ok(vec![NodeTypeDescriptor::Sqrt]),
        OpType::Log => Ok(vec![NodeTypeDescriptor::Ln]),
        OpType::Sign => Ok(vec![NodeTypeDescriptor::Sign]),
        OpType::Reciprocal => Ok(vec![NodeTypeDescriptor::Reciprocal]),
        OpType::Pow => {
            // ONNX Pow 是双输入 (base, exponent)，当 exponent 是常量标量时映射
            // 实际 exponent 值需要在装配层从 initializer 中提取
            // 这里先返回占位，装配层会替换为正确的 exponent
            Ok(vec![NodeTypeDescriptor::Pow { exponent: 1.0 }])
        }

        // ─── Clip ───
        OpType::Clip => {
            let min_val = find_attr_float(attrs, "min").unwrap_or(f32::NEG_INFINITY);
            let max_val = find_attr_float(attrs, "max").unwrap_or(f32::INFINITY);
            if min_val == 0.0 && max_val == 6.0 {
                Ok(vec![NodeTypeDescriptor::ReLU6])
            } else {
                Ok(vec![NodeTypeDescriptor::Clip {
                    min: min_val,
                    max: max_val,
                }])
            }
        }

        // ─── 归约操作 ───
        OpType::ReduceSum => {
            let axis = find_attr_int(attrs, "axes").map(|v| v as usize);
            Ok(vec![NodeTypeDescriptor::Sum { axis }])
        }
        OpType::ReduceMean => {
            let axis = find_attr_int(attrs, "axes").map(|v| v as usize);
            Ok(vec![NodeTypeDescriptor::Mean { axis }])
        }
        OpType::ReduceMax => {
            let axis = find_attr_int(attrs, "axes")
                .ok_or_else(|| OnnxError::UnsupportedAttribute {
                    op_type: "ReduceMax".to_string(),
                    attribute: "axes".to_string(),
                    reason: "需要指定 axis".to_string(),
                })?;
            Ok(vec![NodeTypeDescriptor::Amax {
                axis: axis as usize,
            }])
        }
        OpType::ReduceMin => {
            let axis = find_attr_int(attrs, "axes")
                .ok_or_else(|| OnnxError::UnsupportedAttribute {
                    op_type: "ReduceMin".to_string(),
                    attribute: "axes".to_string(),
                    reason: "需要指定 axis".to_string(),
                })?;
            Ok(vec![NodeTypeDescriptor::Amin {
                axis: axis as usize,
            }])
        }

        // ─── Gemm → MatMul + Add ───
        OpType::Gemm => {
            let alpha = find_attr_float(attrs, "alpha").unwrap_or(1.0);
            let beta = find_attr_float(attrs, "beta").unwrap_or(1.0);
            if (alpha - 1.0).abs() > 1e-6 || (beta - 1.0).abs() > 1e-6 {
                return Err(OnnxError::UnsupportedAttribute {
                    op_type: "Gemm".to_string(),
                    attribute: format!("alpha={alpha}, beta={beta}"),
                    reason: "仅支持 alpha=1.0, beta=1.0（PyTorch nn.Linear 默认导出形式）"
                        .to_string(),
                });
            }
            Ok(vec![NodeTypeDescriptor::MatMul, NodeTypeDescriptor::Add])
        }

        // ─── 张量变换 ───
        OpType::Reshape => Ok(vec![NodeTypeDescriptor::Reshape {
            target_shape: vec![],
        }]),
        OpType::Flatten => {
            let axis = find_attr_int(attrs, "axis").unwrap_or(1);
            if axis != 1 {
                return Err(OnnxError::UnsupportedAttribute {
                    op_type: "Flatten".to_string(),
                    attribute: format!("axis={axis}"),
                    reason: "仅支持 axis=1".to_string(),
                });
            }
            Ok(vec![NodeTypeDescriptor::Flatten {
                keep_first_dim: true,
            }])
        }
        OpType::Concat => {
            let axis = find_attr_int(attrs, "axis").unwrap_or(0);
            Ok(vec![NodeTypeDescriptor::Concat {
                axis: axis as usize,
            }])
        }

        // ─── 正则化 ───
        OpType::Dropout => {
            let ratio = find_attr_float(attrs, "ratio").unwrap_or(0.5);
            Ok(vec![NodeTypeDescriptor::Dropout { p: ratio }])
        }

        // ─── 卷积 ───
        OpType::Conv => {
            validate_conv_config(attrs, node_name)?;
            let strides = find_attr_ints(attrs, "strides");
            let pads = find_attr_ints(attrs, "pads");
            let dilations = find_attr_ints(attrs, "dilations");
            let stride = if strides.len() >= 2 {
                (strides[0] as usize, strides[1] as usize)
            } else {
                (1, 1)
            };
            let padding = if pads.len() >= 4 {
                (pads[0] as usize, pads[2] as usize)
            } else {
                (0, 0)
            };
            let dilation = if dilations.len() >= 2 {
                (dilations[0] as usize, dilations[1] as usize)
            } else {
                (1, 1)
            };
            Ok(vec![NodeTypeDescriptor::Conv2d { stride, padding, dilation }])
        }

        // ─── 转置卷积 ───
        OpType::ConvTranspose => {
            validate_conv_config(attrs, node_name)?;
            let strides = find_attr_ints(attrs, "strides");
            let pads = find_attr_ints(attrs, "pads");
            let output_padding_vals = find_attr_ints(attrs, "output_padding");
            let stride = if strides.len() >= 2 {
                (strides[0] as usize, strides[1] as usize)
            } else {
                (1, 1)
            };
            let padding = if pads.len() >= 4 {
                (pads[0] as usize, pads[2] as usize)
            } else {
                (0, 0)
            };
            let output_padding = if output_padding_vals.len() >= 2 {
                (output_padding_vals[0] as usize, output_padding_vals[1] as usize)
            } else {
                (0, 0)
            };
            Ok(vec![NodeTypeDescriptor::ConvTranspose2d { stride, padding, output_padding }])
        }

        // ─── 池化 ───
        OpType::MaxPool => {
            let kernel = find_attr_ints(attrs, "kernel_shape");
            let strides = find_attr_ints(attrs, "strides");
            if kernel.len() != 2 {
                return Err(OnnxError::UnsupportedConvConfig {
                    op_type: "MaxPool".to_string(),
                    reason: format!("仅支持 2D 池化，kernel_shape 维度={}", kernel.len()),
                });
            }
            let kernel_size = (kernel[0] as usize, kernel[1] as usize);
            let stride = if strides.len() >= 2 {
                (strides[0] as usize, strides[1] as usize)
            } else {
                kernel_size
            };
            Ok(vec![NodeTypeDescriptor::MaxPool2d { kernel_size, stride }])
        }
        OpType::AveragePool => {
            let kernel = find_attr_ints(attrs, "kernel_shape");
            let strides = find_attr_ints(attrs, "strides");
            if kernel.len() != 2 {
                return Err(OnnxError::UnsupportedConvConfig {
                    op_type: "AveragePool".to_string(),
                    reason: format!("仅支持 2D 池化，kernel_shape 维度={}", kernel.len()),
                });
            }
            let kernel_size = (kernel[0] as usize, kernel[1] as usize);
            let stride = if strides.len() >= 2 {
                (strides[0] as usize, strides[1] as usize)
            } else {
                kernel_size
            };
            Ok(vec![NodeTypeDescriptor::AvgPool2d { kernel_size, stride }])
        }

        // ─── 上采样（YOLO PAN/FPN 颈部用）───
        // Resize（opset 13+）和 Upsample（opset 9-10，已废弃但 YOLOv5 老版可能用）
        // 都映射到内部 Upsample2d 节点。
        //
        // 注意：scales 在 ONNX opset 10+ 是从 input initializer 读取的（不在属性里），
        // 这里只能返回占位 scale=0，由装配层（onnx_import.rs）从 initializer
        // 读取实际值后替换（套路与 BatchNormalization::num_features 一致）。
        //
        // 仅支持 mode="nearest"（YOLOv5 用的默认模式）。
        // mode 校验同样推迟到装配层（这里读 attrs.s 后做字符串比较）。
        // TODO[upsample-scales]: 装配层（onnx_import.rs）尚未实现 scales 读取，
        //   完整 ONNX 导入需要等装配层支持后才能端到端跑通。
        OpType::Resize | OpType::Upsample => {
            // 校验 mode 必须是 nearest（属性 mode 是 string 类型，存在 attr.s 里）
            let mode_attr = attrs.iter().find(|a| a.name == "mode");
            if let Some(attr) = mode_attr {
                let mode_str = std::str::from_utf8(attr.s).map_err(|_| OnnxError::UnsupportedAttribute {
                    op_type: "Resize".to_string(),
                    attribute: "mode".to_string(),
                    reason: "mode 属性不是有效 UTF-8".to_string(),
                })?;
                if mode_str != "nearest" {
                    return Err(OnnxError::UnsupportedAttribute {
                        op_type: format!("{op_type:?}"),
                        attribute: "mode".to_string(),
                        reason: format!(
                            "目前仅支持 mode=\"nearest\"，得到 mode=\"{mode_str}\"。\
                            建议用 onnxsim 预处理，或在 PyTorch 端导出时强制 nearest 模式。"
                        ),
                    });
                }
            }
            // 占位 scale=0，装配层负责从 initializer 读取真实值替换
            Ok(vec![NodeTypeDescriptor::Upsample2d {
                scale_h: 0,
                scale_w: 0,
            }])
        }

        // ─── 归一化 ───
        OpType::BatchNormalization => {
            let eps = find_attr_float(attrs, "epsilon").unwrap_or(1e-5);
            let momentum = find_attr_float(attrs, "momentum").unwrap_or(0.1);
            Ok(vec![NodeTypeDescriptor::BatchNormOp {
                eps,
                momentum,
                num_features: 0, // 装配层从输入形状推导
            }])
        }

        // ─── 恒等映射 ───
        OpType::Identity => Ok(vec![NodeTypeDescriptor::Identity]),

        // ─── 逐元素 Min/Max ───
        OpType::Max => Ok(vec![NodeTypeDescriptor::Maximum]),
        OpType::Min => Ok(vec![NodeTypeDescriptor::Minimum]),

        // ─── 不支持的算子 ───
        _ => Err(OnnxError::UnsupportedOperator {
            op_type: op_type.as_str().to_string(),
            node_name: node_name.to_string(),
        }),
    }
}

// ==================== 导出方向：NodeTypeDescriptor → ONNX ====================

/// 导出映射结果：包含 ONNX 算子名和需要设置的属性
pub struct OnnxExportOp {
    /// ONNX 算子名称（如 "Relu"、"Conv" 等）
    pub op_type: &'static str,
    /// 需要设置的浮点属性 (name, value)
    pub float_attrs: Vec<(&'static str, f32)>,
    /// 需要设置的整数属性 (name, value)
    pub int_attrs: Vec<(&'static str, i64)>,
    /// 需要设置的整数列表属性 (name, values)
    pub int_list_attrs: Vec<(&'static str, Vec<i64>)>,
}

impl OnnxExportOp {
    fn simple(op_type: &'static str) -> Self {
        Self {
            op_type,
            float_attrs: vec![],
            int_attrs: vec![],
            int_list_attrs: vec![],
        }
    }
}

/// 节点在导出方向上的分类
pub enum ExportCategory {
    /// 可导出为 ONNX 算子节点
    Operator(OnnxExportOp),
    /// 图输入节点（映射为 ONNX graph input）
    GraphInput,
    /// 参数节点（映射为 ONNX initializer）
    Initializer,
    /// 状态节点（映射为 ONNX initializer，初值为零张量）
    StateInitializer,
    /// 训练专用节点，不可导出到推理图
    TrainingOnly,
    /// 不支持导出的节点类型
    Unsupported(String),
}

/// 将 NodeTypeDescriptor 映射为导出分类。
pub fn descriptor_to_export_category(desc: &NodeTypeDescriptor) -> ExportCategory {
    match desc {
        // ─── 图输入 ───
        NodeTypeDescriptor::BasicInput => ExportCategory::GraphInput,

        // ─── 参数 / 状态 ───
        NodeTypeDescriptor::Parameter => ExportCategory::Initializer,
        NodeTypeDescriptor::State => ExportCategory::StateInitializer,

        // ─── 训练专用（不可导出） ───
        NodeTypeDescriptor::TargetInput
        | NodeTypeDescriptor::SoftmaxCrossEntropy
        | NodeTypeDescriptor::BCE { .. }
        | NodeTypeDescriptor::MSE { .. }
        | NodeTypeDescriptor::MAE { .. }
        | NodeTypeDescriptor::Huber { .. } => ExportCategory::TrainingOnly,

        // ─── 激活函数 ───
        NodeTypeDescriptor::ReLU => ExportCategory::Operator(OnnxExportOp::simple("Relu")),
        NodeTypeDescriptor::Sigmoid => ExportCategory::Operator(OnnxExportOp::simple("Sigmoid")),
        NodeTypeDescriptor::Tanh => ExportCategory::Operator(OnnxExportOp::simple("Tanh")),
        NodeTypeDescriptor::Softmax => ExportCategory::Operator(OnnxExportOp::simple("Softmax")),
        NodeTypeDescriptor::LogSoftmax => {
            ExportCategory::Operator(OnnxExportOp::simple("LogSoftmax"))
        }
        NodeTypeDescriptor::Gelu => ExportCategory::Operator(OnnxExportOp::simple("Gelu")),
        NodeTypeDescriptor::Selu => ExportCategory::Operator(OnnxExportOp::simple("Selu")),
        NodeTypeDescriptor::Mish => ExportCategory::Operator(OnnxExportOp::simple("Mish")),
        NodeTypeDescriptor::HardSwish => {
            ExportCategory::Operator(OnnxExportOp::simple("HardSwish"))
        }
        NodeTypeDescriptor::HardSigmoid => {
            ExportCategory::Operator(OnnxExportOp::simple("HardSigmoid"))
        }
        NodeTypeDescriptor::SoftPlus => ExportCategory::Operator(OnnxExportOp::simple("Softplus")),
        NodeTypeDescriptor::Swish => {
            // Swish = x * sigmoid(x)，ONNX 无原生支持，标记为不支持
            ExportCategory::Unsupported("Swish（ONNX 无原生算子，需子图展开）".to_string())
        }
        NodeTypeDescriptor::Elu { alpha } => ExportCategory::Operator(OnnxExportOp {
            op_type: "Elu",
            float_attrs: vec![("alpha", *alpha)],
            int_attrs: vec![],
            int_list_attrs: vec![],
        }),
        NodeTypeDescriptor::LeakyReLU { alpha } => ExportCategory::Operator(OnnxExportOp {
            op_type: "LeakyRelu",
            float_attrs: vec![("alpha", *alpha)],
            int_attrs: vec![],
            int_list_attrs: vec![],
        }),
        NodeTypeDescriptor::ReLU6 => ExportCategory::Operator(OnnxExportOp {
            op_type: "Clip",
            float_attrs: vec![("min", 0.0), ("max", 6.0)],
            int_attrs: vec![],
            int_list_attrs: vec![],
        }),
        NodeTypeDescriptor::HardTanh { min_val, max_val } => {
            ExportCategory::Operator(OnnxExportOp {
                op_type: "Clip",
                float_attrs: vec![("min", *min_val), ("max", *max_val)],
                int_attrs: vec![],
                int_list_attrs: vec![],
            })
        }

        // ─── 算术运算 ───
        NodeTypeDescriptor::Add => ExportCategory::Operator(OnnxExportOp::simple("Add")),
        NodeTypeDescriptor::Subtract => ExportCategory::Operator(OnnxExportOp::simple("Sub")),
        NodeTypeDescriptor::Multiply => ExportCategory::Operator(OnnxExportOp::simple("Mul")),
        NodeTypeDescriptor::Divide => ExportCategory::Operator(OnnxExportOp::simple("Div")),
        NodeTypeDescriptor::Negate => ExportCategory::Operator(OnnxExportOp::simple("Neg")),
        NodeTypeDescriptor::MatMul => ExportCategory::Operator(OnnxExportOp::simple("MatMul")),

        // ─── 数学运算 ───
        NodeTypeDescriptor::Abs => ExportCategory::Operator(OnnxExportOp::simple("Abs")),
        NodeTypeDescriptor::Exp => ExportCategory::Operator(OnnxExportOp::simple("Exp")),
        NodeTypeDescriptor::Sqrt => ExportCategory::Operator(OnnxExportOp::simple("Sqrt")),
        NodeTypeDescriptor::Ln => ExportCategory::Operator(OnnxExportOp::simple("Log")),
        NodeTypeDescriptor::Sign => ExportCategory::Operator(OnnxExportOp::simple("Sign")),
        NodeTypeDescriptor::Reciprocal => {
            ExportCategory::Operator(OnnxExportOp::simple("Reciprocal"))
        }
        NodeTypeDescriptor::Square => ExportCategory::Operator(OnnxExportOp {
            op_type: "Pow",
            float_attrs: vec![],
            int_attrs: vec![],
            int_list_attrs: vec![],
        }),
        NodeTypeDescriptor::Pow { .. } => ExportCategory::Operator(OnnxExportOp::simple("Pow")),
        NodeTypeDescriptor::Log10 => {
            ExportCategory::Unsupported("Log10（ONNX 仅有自然对数 Log）".to_string())
        }
        NodeTypeDescriptor::Log2 => {
            ExportCategory::Unsupported("Log2（ONNX 仅有自然对数 Log）".to_string())
        }

        // ─── 归约 ───
        NodeTypeDescriptor::Sum { axis } => ExportCategory::Operator(OnnxExportOp {
            op_type: "ReduceSum",
            float_attrs: vec![],
            int_attrs: vec![],
            int_list_attrs: if let Some(a) = axis {
                vec![("axes", vec![*a as i64])]
            } else {
                vec![]
            },
        }),
        NodeTypeDescriptor::Mean { axis } => ExportCategory::Operator(OnnxExportOp {
            op_type: "ReduceMean",
            float_attrs: vec![],
            int_attrs: vec![],
            int_list_attrs: if let Some(a) = axis {
                vec![("axes", vec![*a as i64])]
            } else {
                vec![]
            },
        }),
        NodeTypeDescriptor::Amax { axis } => ExportCategory::Operator(OnnxExportOp {
            op_type: "ReduceMax",
            float_attrs: vec![],
            int_attrs: vec![],
            int_list_attrs: vec![("axes", vec![*axis as i64])],
        }),
        NodeTypeDescriptor::Amin { axis } => ExportCategory::Operator(OnnxExportOp {
            op_type: "ReduceMin",
            float_attrs: vec![],
            int_attrs: vec![],
            int_list_attrs: vec![("axes", vec![*axis as i64])],
        }),

        // ─── 逐元素 Min/Max ───
        NodeTypeDescriptor::Maximum => ExportCategory::Operator(OnnxExportOp::simple("Max")),
        NodeTypeDescriptor::Minimum => ExportCategory::Operator(OnnxExportOp::simple("Min")),

        // ─── 张量变换 ───
        NodeTypeDescriptor::Reshape { .. } => {
            ExportCategory::Operator(OnnxExportOp::simple("Reshape"))
        }
        NodeTypeDescriptor::Flatten { .. } => ExportCategory::Operator(OnnxExportOp {
            op_type: "Flatten",
            float_attrs: vec![],
            int_attrs: vec![("axis", 1)],
            int_list_attrs: vec![],
        }),
        NodeTypeDescriptor::Concat { axis } => ExportCategory::Operator(OnnxExportOp {
            op_type: "Concat",
            float_attrs: vec![],
            int_attrs: vec![("axis", *axis as i64)],
            int_list_attrs: vec![],
        }),
        NodeTypeDescriptor::Identity => ExportCategory::Operator(OnnxExportOp::simple("Identity")),
        NodeTypeDescriptor::Clip { min, max } => ExportCategory::Operator(OnnxExportOp {
            op_type: "Clip",
            float_attrs: vec![("min", *min), ("max", *max)],
            int_attrs: vec![],
            int_list_attrs: vec![],
        }),

        // ─── 正则化 ───
        NodeTypeDescriptor::Dropout { p } => ExportCategory::Operator(OnnxExportOp {
            op_type: "Dropout",
            float_attrs: vec![("ratio", *p)],
            int_attrs: vec![],
            int_list_attrs: vec![],
        }),

        // ─── 卷积 / 池化 ───
        NodeTypeDescriptor::Conv2d { stride, padding, dilation } => {
            let mut attrs = vec![
                ("strides", vec![stride.0 as i64, stride.1 as i64]),
                (
                    "pads",
                    vec![
                        padding.0 as i64,
                        padding.1 as i64,
                        padding.0 as i64,
                        padding.1 as i64,
                    ],
                ),
            ];
            if *dilation != (1, 1) {
                attrs.push(("dilations", vec![dilation.0 as i64, dilation.1 as i64]));
            }
            ExportCategory::Operator(OnnxExportOp {
                op_type: "Conv",
                float_attrs: vec![],
                int_attrs: vec![],
                int_list_attrs: attrs,
            })
        }
        NodeTypeDescriptor::ConvTranspose2d { stride, padding, output_padding } => {
            let mut attrs = vec![
                ("strides", vec![stride.0 as i64, stride.1 as i64]),
                (
                    "pads",
                    vec![
                        padding.0 as i64,
                        padding.1 as i64,
                        padding.0 as i64,
                        padding.1 as i64,
                    ],
                ),
            ];
            if *output_padding != (0, 0) {
                attrs.push(("output_padding", vec![output_padding.0 as i64, output_padding.1 as i64]));
            }
            ExportCategory::Operator(OnnxExportOp {
                op_type: "ConvTranspose",
                float_attrs: vec![],
                int_attrs: vec![],
                int_list_attrs: attrs,
            })
        }
        NodeTypeDescriptor::MaxPool2d { kernel_size, stride } => {
            ExportCategory::Operator(OnnxExportOp {
                op_type: "MaxPool",
                float_attrs: vec![],
                int_attrs: vec![],
                int_list_attrs: vec![
                    (
                        "kernel_shape",
                        vec![kernel_size.0 as i64, kernel_size.1 as i64],
                    ),
                    ("strides", vec![stride.0 as i64, stride.1 as i64]),
                ],
            })
        }
        NodeTypeDescriptor::AvgPool2d { kernel_size, stride } => {
            ExportCategory::Operator(OnnxExportOp {
                op_type: "AveragePool",
                float_attrs: vec![],
                int_attrs: vec![],
                int_list_attrs: vec![
                    (
                        "kernel_shape",
                        vec![kernel_size.0 as i64, kernel_size.1 as i64],
                    ),
                    ("strides", vec![stride.0 as i64, stride.1 as i64]),
                ],
            })
        }

        // 上采样：导出为 ONNX Resize（opset 13 nearest）
        // 注意：完整 Resize 算子需要 string 属性 mode="nearest" 和 input "scales"，
        // 当前 OnnxExportOp 仅支持 float/int/int_list 三种属性类型，
        // 因此这里返回 simple("Resize") 占位（不带任何属性），
        // 与策略文档 §4.4 一致——only_torch 不承诺 ONNX 导出 round-trip。
        // TODO[upsample-export]: 后续若需要完整导出，需扩展 OnnxExportOp 支持
        //   string_attrs 和 float_list_attrs，并由调用方写入 scales initializer。
        NodeTypeDescriptor::Upsample2d { .. } => {
            ExportCategory::Operator(OnnxExportOp::simple("Resize"))
        }

        // ─── 归一化 ───
        NodeTypeDescriptor::BatchNormOp { eps, momentum, .. } => {
            ExportCategory::Operator(OnnxExportOp {
                op_type: "BatchNormalization",
                float_attrs: vec![("epsilon", *eps), ("momentum", *momentum)],
                int_attrs: vec![],
                int_list_attrs: vec![],
            })
        }

        // ─── 循环单元 ───
        NodeTypeDescriptor::CellRnn { .. } => ExportCategory::Operator(OnnxExportOp::simple("RNN")),
        NodeTypeDescriptor::CellLstm { .. } => {
            ExportCategory::Operator(OnnxExportOp::simple("LSTM"))
        }
        NodeTypeDescriptor::CellGru { .. } => ExportCategory::Operator(OnnxExportOp::simple("GRU")),

        // ─── 零张量 ───
        NodeTypeDescriptor::ZerosLike => {
            ExportCategory::Unsupported("ZerosLike（内部辅助节点）".to_string())
        }

        // ─── 梯度屏障 ───
        NodeTypeDescriptor::Detach => ExportCategory::Operator(OnnxExportOp::simple("Identity")),

        // ─── 其他不支持导出的节点 ───
        NodeTypeDescriptor::Step
        | NodeTypeDescriptor::Select { .. }
        | NodeTypeDescriptor::Gather { .. }
        | NodeTypeDescriptor::Stack { .. }
        | NodeTypeDescriptor::WhereCond { .. }
        | NodeTypeDescriptor::Narrow { .. }
        | NodeTypeDescriptor::Permute { .. }
        | NodeTypeDescriptor::Pad { .. }
        | NodeTypeDescriptor::Repeat { .. }
        | NodeTypeDescriptor::TopK { .. }
        | NodeTypeDescriptor::SortNode { .. }
        | NodeTypeDescriptor::LayerNormOp { .. }
        | NodeTypeDescriptor::RMSNormOp { .. } => {
            ExportCategory::Unsupported(format!("{desc:?}"))
        }
    }
}

// ==================== 辅助函数 ====================

/// 从属性列表中查找指定名称的浮点属性
pub(crate) fn find_attr_float(attrs: &[Attribute], name: &str) -> Option<f32> {
    attrs.iter().find(|a| a.name == name).map(|a| a.f)
}

/// 从属性列表中查找指定名称的整数属性
pub(crate) fn find_attr_int(attrs: &[Attribute], name: &str) -> Option<i64> {
    attrs.iter().find(|a| a.name == name).map(|a| a.i)
}

/// 从属性列表中查找指定名称的整数列表属性
pub(crate) fn find_attr_ints(attrs: &[Attribute], name: &str) -> Vec<i64> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| a.ints.clone())
        .unwrap_or_default()
}

/// 验证 Conv 配置（仅支持 2D、group=1、dilation=1）
fn validate_conv_config(attrs: &[Attribute], node_name: &str) -> Result<(), OnnxError> {
    let kernel = find_attr_ints(attrs, "kernel_shape");
    if !kernel.is_empty() && kernel.len() != 2 {
        return Err(OnnxError::UnsupportedConvConfig {
            op_type: "Conv".to_string(),
            reason: format!(
                "仅支持 2D 卷积，kernel_shape 维度={} (节点: \"{node_name}\")",
                kernel.len()
            ),
        });
    }
    let group = find_attr_int(attrs, "group").unwrap_or(1);
    if group != 1 {
        return Err(OnnxError::UnsupportedConvConfig {
            op_type: "Conv".to_string(),
            reason: format!(
                "不支持分组卷积 group={group} (节点: \"{node_name}\")"
            ),
        });
    }
    let dilations = find_attr_ints(attrs, "dilations");
    if !dilations.is_empty() && dilations.len() != 2 {
        return Err(OnnxError::UnsupportedConvConfig {
            op_type: "Conv".to_string(),
            reason: format!(
                "仅支持 2D 空洞卷积，dilations 维度={} (节点: \"{node_name}\")",
                dilations.len()
            ),
        });
    }
    Ok(())
}
