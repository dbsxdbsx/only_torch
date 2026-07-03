use std::fs::File;
use std::io::{Read, Write};

use super::{Tensor, next_source_id};

// 保存和加载张量
impl Tensor {
    /// 将单个Tensor写入本地文件
    pub fn save(&self, file: &mut File) {
        let serialized_data = bincode::serialize(&self.data).unwrap();
        file.write_all(&serialized_data).unwrap();
    }
    /// 从本地文件加载单个Tensor
    pub fn load(file: &mut File) -> Self {
        let mut serialized_data = Vec::new();
        file.read_to_end(&mut serialized_data).unwrap();
        // 序列化格式与旧 `ArrayD`（OwnedRepr）完全兼容：ndarray 的 serde 只编码 dim + data，
        // 与存储 repr 无关，故旧文件可直接反序列化为 ArcArray。
        let data: super::TensorStorage = bincode::deserialize(&serialized_data).unwrap();
        Self {
            data,
            source_id: next_source_id(),
        }
    }
}
