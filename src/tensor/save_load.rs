use std::fs::File;
use std::io::{Read, Write};

use super::Tensor;

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
        let data = bincode::deserialize(&serialized_data).unwrap();
        Self { data }
    }
}
