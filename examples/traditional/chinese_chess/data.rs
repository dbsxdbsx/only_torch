//! 中国象棋训练数据加载器
//!
//! 分别加载 `train/` 和 `test/` 子目录的二进制数据，
//! 确保训练集和测试集使用不同的视觉风格（字体、配色）。
//!
//! 支持可选的真实棋子数据混合（`data/chinese_chess_real/`），
//! 与 PyTorch 版 `--real-data` 功能对齐。
//!
//! 数据格式（由 `scripts/generate_chess_data.py` 生成）：
//! - `images.bin`: header [u32 N, C, H, W] + N*C*H*W f32 values
//! - `labels.bin`: header [u32 N] + N u8 values (类别 0-14)

use only_torch::tensor::Tensor;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// 15 类类别名称
pub const CLASS_NAMES: [&str; 15] = [
    "空位", "红帅", "红仕", "红相", "红車", "红馬", "红炮", "红兵",
    "黑将", "黑士", "黑象", "黑車", "黑馬", "黑炮", "黑卒",
];

/// 加载训练集和测试集（合成 + 可选真实数据）
///
/// 从 `data_dir/train/` 和 `data_dir/test/` 加载合成数据，
/// 若 `real_data_dir` 存在则混入真实棋子数据。
///
/// 返回 `((train_images, train_labels), (test_images, test_labels))`
/// - images: [N, C, H, W] float32
/// - labels: [N, 15] one-hot float32
pub fn load_chess_data(
    data_dir: &str,
) -> Result<((Tensor, Tensor), (Tensor, Tensor)), String> {
    let real_data_dir = "data/chinese_chess_real";
    let has_real = Path::new(real_data_dir).join("train").join("images.bin").exists();

    // 加载合成数据
    let train_dir = Path::new(data_dir).join("train");
    let test_dir = Path::new(data_dir).join("test");

    println!("  加载合成训练集: {}", train_dir.display());
    let (train_images, train_labels) = load_split(&train_dir)?;
    println!(
        "    合成训练集: {} 样本",
        train_images.shape()[0],
    );

    println!("  加载合成测试集: {}", test_dir.display());
    let (test_images, test_labels) = load_split(&test_dir)?;
    println!(
        "    合成测试集: {} 样本",
        test_images.shape()[0],
    );

    // 混入真实数据（如果存在）
    if has_real {
        let real_train_dir = Path::new(real_data_dir).join("train");
        let real_test_dir = Path::new(real_data_dir).join("test");

        println!("  加载真实训练集: {}", real_train_dir.display());
        let (real_train_img, real_train_lbl) = load_split(&real_train_dir)?;
        println!("    真实训练集: {} 样本", real_train_img.shape()[0]);

        let (merged_test_img, merged_test_lbl) = if real_test_dir.join("images.bin").exists() {
            println!("  加载真实测试集: {}", real_test_dir.display());
            let (real_test_img, real_test_lbl) = load_split(&real_test_dir)?;
            println!("    真实测试集: {} 样本", real_test_img.shape()[0]);
            (concat_tensors(&test_images, &real_test_img), concat_tensors(&test_labels, &real_test_lbl))
        } else {
            (test_images, test_labels)
        };

        let merged_train_img = concat_tensors(&train_images, &real_train_img);
        let merged_train_lbl = concat_tensors(&train_labels, &real_train_lbl);

        println!(
            "  合并后: 训练 {} 样本, 测试 {} 样本",
            merged_train_img.shape()[0],
            merged_test_img.shape()[0],
        );

        Ok(((merged_train_img, merged_train_lbl), (merged_test_img, merged_test_lbl)))
    } else {
        Ok(((train_images, train_labels), (test_images, test_labels)))
    }
}

/// 加载单个 split 的二进制数据
fn load_split(split_dir: &Path) -> Result<(Tensor, Tensor), String> {
    let images_path = split_dir.join("images.bin");
    let labels_path = split_dir.join("labels.bin");

    if !images_path.exists() {
        return Err(format!(
            "数据文件不存在: {}\n请先运行: python scripts/generate_chess_data.py",
            images_path.display()
        ));
    }

    // 读取图像
    let mut img_file =
        File::open(&images_path).map_err(|e| format!("打开 images.bin 失败: {e}"))?;

    let mut header = [0u8; 16];
    img_file
        .read_exact(&mut header)
        .map_err(|e| format!("读取图像 header 失败: {e}"))?;

    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let c = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let h = u32::from_le_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let w = u32::from_le_bytes([header[12], header[13], header[14], header[15]]) as usize;

    let num_floats = n * c * h * w;
    let mut img_bytes = vec![0u8; num_floats * 4];
    img_file
        .read_exact(&mut img_bytes)
        .map_err(|e| format!("读取图像数据失败: {e}"))?;

    let img_data: Vec<f32> = img_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let images = Tensor::new(&img_data, &[n, c, h, w]);

    // 读取标签
    let mut lbl_file =
        File::open(&labels_path).map_err(|e| format!("打开 labels.bin 失败: {e}"))?;

    let mut lbl_header = [0u8; 4];
    lbl_file
        .read_exact(&mut lbl_header)
        .map_err(|e| format!("读取标签 header 失败: {e}"))?;

    let n_labels = u32::from_le_bytes(lbl_header) as usize;
    assert_eq!(n, n_labels, "图像数量与标签数量不匹配");

    let mut label_bytes = vec![0u8; n_labels];
    lbl_file
        .read_exact(&mut label_bytes)
        .map_err(|e| format!("读取标签数据失败: {e}"))?;

    // 转为 one-hot 编码 [N, 15]
    let num_classes = 15;
    let mut one_hot = vec![0.0f32; n * num_classes];
    for (i, &label) in label_bytes.iter().enumerate() {
        one_hot[i * num_classes + label as usize] = 1.0;
    }
    let labels = Tensor::new(&one_hot, &[n, num_classes]);

    Ok((images, labels))
}

/// 沿 batch 维度拼接两个张量
fn concat_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor::concat(&[a, b], 0)
}
