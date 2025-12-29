//! MNIST 手写数字数据集
//!
//! 支持：
//! - IDX 二进制格式解析（支持 .gz 压缩）
//! - 像素归一化 (0-255 → 0-1)
//! - 标签 one-hot 编码
//! - 可选自动下载

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;

use crate::data::error::DataError;
use crate::data::transforms::{normalize_pixels, one_hot};
use crate::tensor::Tensor;

/// MNIST 下载地址（使用 AWS S3 镜像，原官网 yann.lecun.com 不稳定）
const MNIST_BASE_URL: &str = "https://ossci-datasets.s3.amazonaws.com/mnist/";

/// MNIST 文件信息
#[allow(dead_code)]
const MNIST_FILES: [(&str, &str); 4] = [
    (
        "train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    (
        "train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    (
        "t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    (
        "t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
];

/// MNIST 手写数字数据集
///
/// 包含 60,000 个训练样本和 10,000 个测试样本。
/// 每个样本是 28x28 的灰度图像，标签为 0-9。
#[derive(Debug, Clone)]
pub struct MnistDataset {
    /// 图像数据 [N, 1, 28, 28] 或 flatten 后 [N, 784]
    images: Tensor,
    /// 标签数据 [N, 10] (one-hot)
    labels: Tensor,
    /// 样本数量
    len: usize,
    /// 是否已展平
    is_flattened: bool,
}

impl MnistDataset {
    /// 完整加载 API
    ///
    /// # 参数
    /// - `root`: 数据目录，None 则使用默认 (~/.cache/only_torch/datasets/mnist)
    /// - `train`: true=训练集(60000), false=测试集(10000)
    /// - `download`: true=自动下载缺失文件
    ///
    /// # 返回
    /// 加载后的 MnistDataset，图像形状为 [N, 1, 28, 28]
    pub fn load(root: Option<&str>, train: bool, download: bool) -> Result<Self, DataError> {
        let data_dir = root
            .map(PathBuf::from)
            .unwrap_or_else(|| default_data_dir().join("mnist"));

        // 确定文件名
        let (images_file, labels_file) = if train {
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        } else {
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        };

        // 检查文件是否存在，必要时下载
        let images_path = ensure_file(&data_dir, images_file, download)?;
        let labels_path = ensure_file(&data_dir, labels_file, download)?;

        // 解析 IDX 文件
        let images_raw = parse_idx_images(&images_path)?;
        let labels_raw = parse_idx_labels(&labels_path)?;

        let len = labels_raw.shape()[0];

        // 归一化像素值 [0, 255] -> [0, 1]
        let images_normalized = normalize_pixels(&images_raw);

        // 重塑为 [N, 1, 28, 28] (NCHW 格式)
        let images = images_normalized.reshape(&[len, 1, 28, 28]);

        // one-hot 编码标签
        let labels = one_hot(&labels_raw, 10);

        Ok(Self {
            images,
            labels,
            len,
            is_flattened: false,
        })
    }

    /// 便捷 API：加载训练集（默认路径，自动下载）
    pub fn train() -> Result<Self, DataError> {
        Self::load(None, true, true)
    }

    /// 便捷 API：加载测试集（默认路径，自动下载）
    pub fn test() -> Result<Self, DataError> {
        Self::load(None, false, true)
    }

    /// 将图像展平为 [N, 784]（用于 MLP）
    ///
    /// 消耗 self，返回新的展平版本
    pub fn flatten(mut self) -> Self {
        if !self.is_flattened {
            self.images = self.images.reshape(&[self.len, 784]);
            self.is_flattened = true;
        }
        self
    }

    /// 返回数据集中的样本数量
    pub fn len(&self) -> usize {
        self.len
    }

    /// 数据集是否为空
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 获取第 index 个样本
    ///
    /// # 返回
    /// (image, label) 元组
    /// - image: [1, 28, 28] 或 [784]（若已 flatten）
    /// - label: [10] (one-hot)
    pub fn get(&self, index: usize) -> Result<(Tensor, Tensor), DataError> {
        if index >= self.len {
            return Err(DataError::IndexOutOfBounds {
                index,
                len: self.len,
            });
        }

        let image = if self.is_flattened {
            // [N, 784] -> [784]
            self.images.slice(&[&index, &(..)]).flatten()
        } else {
            // [N, 1, 28, 28] -> [1, 28, 28]
            self.images
                .slice(&[&index, &(..), &(..), &(..)])
                .reshape(&[1, 28, 28])
        };

        // [N, 10] -> [10]
        let label = self.labels.slice(&[&index, &(..)]).flatten();

        Ok((image, label))
    }

    /// 输入的形状（不含 batch 维度）
    pub fn input_shape(&self) -> Vec<usize> {
        if self.is_flattened {
            vec![784]
        } else {
            vec![1, 28, 28]
        }
    }

    /// 标签的形状（不含 batch 维度）
    pub fn label_shape(&self) -> Vec<usize> {
        vec![10]
    }

    /// 获取所有图像（用于批量处理）
    pub fn images(&self) -> &Tensor {
        &self.images
    }

    /// 获取所有标签（用于批量处理）
    pub fn labels(&self) -> &Tensor {
        &self.labels
    }
}

/// 获取默认数据目录
pub fn default_data_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("only_torch")
        .join("datasets")
}

/// 确保文件存在，必要时下载
fn ensure_file(data_dir: &Path, base_name: &str, download: bool) -> Result<PathBuf, DataError> {
    // 优先检查解压后的文件
    let uncompressed_path = data_dir.join(base_name);
    if uncompressed_path.exists() {
        return Ok(uncompressed_path);
    }

    // 检查 .gz 文件
    let gz_path = data_dir.join(format!("{}.gz", base_name));
    if gz_path.exists() {
        return Ok(gz_path);
    }

    // 文件不存在，尝试下载
    if download {
        std::fs::create_dir_all(data_dir).map_err(DataError::IoError)?;
        download_file(base_name, &gz_path)?;
        Ok(gz_path)
    } else {
        Err(DataError::FileNotFound(uncompressed_path))
    }
}

/// 下载 MNIST 文件
fn download_file(base_name: &str, dest_path: &Path) -> Result<(), DataError> {
    let gz_name = format!("{}.gz", base_name);
    let url = format!("{}{}", MNIST_BASE_URL, gz_name);

    println!("正在下载 {} ...", url);

    let response = ureq::get(&url)
        .call()
        .map_err(|e| DataError::DownloadError(format!("HTTP 请求失败: {}", e)))?;

    if response.status() != 200 {
        return Err(DataError::DownloadError(format!(
            "HTTP 状态码: {}",
            response.status()
        )));
    }

    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| DataError::DownloadError(format!("读取响应失败: {}", e)))?;

    // 验证 MD5（可选，暂时跳过详细实现）
    // TODO: 添加 MD5 校验

    std::fs::write(dest_path, &bytes).map_err(DataError::IoError)?;

    println!("下载完成: {:?}", dest_path);
    Ok(())
}

/// 解析 IDX 图像文件
///
/// IDX 格式：
/// - [0-3] magic number (0x00000803 = 2051)
/// - [4-7] number of images
/// - [8-11] number of rows
/// - [12-15] number of columns
/// - [16+] pixel data (unsigned byte)
fn parse_idx_images(path: &Path) -> Result<Tensor, DataError> {
    let file = File::open(path).map_err(|_| DataError::FileNotFound(path.to_path_buf()))?;
    let reader: Box<dyn Read> = if path.extension().map_or(false, |ext| ext == "gz") {
        Box::new(GzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut reader = reader;
    let mut header = [0u8; 16];
    reader
        .read_exact(&mut header)
        .map_err(|e| DataError::FormatError(format!("读取头部失败: {}", e)))?;

    // 解析头部（大端序）
    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if magic != 2051 {
        return Err(DataError::FormatError(format!(
            "无效的 magic number: {} (期望 2051)",
            magic
        )));
    }

    let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;
    let num_rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]) as usize;
    let num_cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]) as usize;

    if num_rows != 28 || num_cols != 28 {
        return Err(DataError::FormatError(format!(
            "无效的图像尺寸: {}x{} (期望 28x28)",
            num_rows, num_cols
        )));
    }

    // 读取像素数据
    let pixel_count = num_images * 28 * 28;
    let mut pixels = vec![0u8; pixel_count];
    reader
        .read_exact(&mut pixels)
        .map_err(|e| DataError::FormatError(format!("读取像素数据失败: {}", e)))?;

    // 转换为 f32 Tensor [N, 784]
    let data: Vec<f32> = pixels.into_iter().map(|p| p as f32).collect();
    Ok(Tensor::new(&data, &[num_images, 784]))
}

/// 解析 IDX 标签文件
///
/// IDX 格式：
/// - [0-3] magic number (0x00000801 = 2049)
/// - [4-7] number of labels
/// - [8+] label data (unsigned byte, 0-9)
fn parse_idx_labels(path: &Path) -> Result<Tensor, DataError> {
    let file = File::open(path).map_err(|_| DataError::FileNotFound(path.to_path_buf()))?;
    let reader: Box<dyn Read> = if path.extension().map_or(false, |ext| ext == "gz") {
        Box::new(GzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut reader = reader;
    let mut header = [0u8; 8];
    reader
        .read_exact(&mut header)
        .map_err(|e| DataError::FormatError(format!("读取头部失败: {}", e)))?;

    // 解析头部（大端序）
    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    if magic != 2049 {
        return Err(DataError::FormatError(format!(
            "无效的 magic number: {} (期望 2049)",
            magic
        )));
    }

    let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]) as usize;

    // 读取标签数据
    let mut labels = vec![0u8; num_labels];
    reader
        .read_exact(&mut labels)
        .map_err(|e| DataError::FormatError(format!("读取标签数据失败: {}", e)))?;

    // 转换为 f32 Tensor [N]
    let data: Vec<f32> = labels.into_iter().map(|l| l as f32).collect();
    Ok(Tensor::new(&data, &[num_labels]))
}
