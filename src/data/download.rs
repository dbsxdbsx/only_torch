//! 通用下载工具
//!
//! 提供 HTTP 下载和 MD5 校验功能，供各数据集复用。

use std::io::Read;
use std::path::Path;

use md5::{Digest, Md5};

use super::error::DataError;

/// 下载文件并保存到指定路径
///
/// # 参数
/// - `url`: 下载地址
/// - `dest_path`: 保存路径
/// - `expected_md5`: 可选的 MD5 校验码，提供时会验证下载内容
///
/// # 返回
/// - 成功返回 `Ok(())`
/// - 失败返回 `DataError::DownloadError`
pub fn download_file(
    url: &str,
    dest_path: &Path,
    expected_md5: Option<&str>,
) -> Result<(), DataError> {
    println!("正在下载 {url} ...");

    let response = ureq::get(url)
        .call()
        .map_err(|e| DataError::DownloadError(format!("HTTP 请求失败: {e}")))?;

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
        .map_err(|e| DataError::DownloadError(format!("读取响应失败: {e}")))?;

    // MD5 校验（如果提供了预期值）
    if let Some(expected) = expected_md5 {
        let actual = compute_md5(&bytes);
        if actual != expected {
            return Err(DataError::DownloadError(format!(
                "MD5 校验失败: 期望 {expected}, 实际 {actual}"
            )));
        }
        println!("MD5 校验通过: {actual}");
    }

    std::fs::write(dest_path, &bytes).map_err(DataError::IoError)?;

    println!("下载完成: {dest_path:?}");
    Ok(())
}

/// 下载文本内容并保存到指定路径
///
/// 适用于 CSV 等文本格式数据集。
///
/// # 参数
/// - `url`: 下载地址
/// - `dest_path`: 保存路径
/// - `expected_md5`: 可选的 MD5 校验码
pub fn download_text(
    url: &str,
    dest_path: &Path,
    expected_md5: Option<&str>,
) -> Result<(), DataError> {
    println!("正在下载 {url} ...");

    let response = ureq::get(url)
        .call()
        .map_err(|e| DataError::DownloadError(format!("HTTP 请求失败: {e}")))?;

    if response.status() != 200 {
        return Err(DataError::DownloadError(format!(
            "HTTP 状态码: {}",
            response.status()
        )));
    }

    let mut content = String::new();
    response
        .into_reader()
        .read_to_string(&mut content)
        .map_err(|e| DataError::DownloadError(format!("读取响应失败: {e}")))?;

    // MD5 校验（如果提供了预期值）
    if let Some(expected) = expected_md5 {
        let actual = compute_md5(content.as_bytes());
        if actual != expected {
            return Err(DataError::DownloadError(format!(
                "MD5 校验失败: 期望 {expected}, 实际 {actual}"
            )));
        }
        println!("MD5 校验通过: {actual}");
    }

    std::fs::write(dest_path, &content).map_err(DataError::IoError)?;

    println!("下载完成: {dest_path:?}");
    Ok(())
}

/// 计算数据的 MD5 校验码
pub fn compute_md5(data: &[u8]) -> String {
    let mut hasher = Md5::new();
    hasher.update(data);
    let result = hasher.finalize();
    format!("{:x}", result)
}
