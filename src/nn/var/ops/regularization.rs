/*
 * @Author       : 老董
 * @Date         : 2026-01-29
 * @Description  : Var 正则化扩展 trait
 *
 * 提供正则化操作的链式调用支持。
 */

use crate::nn::{GraphError, Var};
use std::rc::Rc;

/// 正则化扩展 trait
///
/// 提供常用正则化操作的链式调用：
/// - `dropout(p)`: 以概率 p 随机丢弃元素
///
/// # 使用示例
/// ```ignore
/// use only_torch::nn::{Var, VarRegularizationOps, DEFAULT_DROPOUT_P};
///
/// // 训练时随机丢弃 30%
/// let h = x.dropout(0.3)?;
///
/// // 使用推荐默认值 0.5
/// let h = x.dropout(DEFAULT_DROPOUT_P)?;
/// ```
///
/// # 注意
/// - Dropout 仅在训练模式下生效，评估模式下直接通过
/// - 使用 `graph.train()` / `graph.eval()` 切换模式
pub trait VarRegularizationOps {
    /// Dropout 正则化
    ///
    /// 训练时以概率 p 随机丢弃元素（Inverted Dropout），评估时直接通过。
    ///
    /// # 参数
    /// - `p`: 丢弃概率，范围 [0.0, 1.0)
    ///
    /// # 错误
    /// - `p` 不在 [0, 1) 范围内
    ///
    /// # 推荐值
    /// - 全连接层：0.5（经典值）
    /// - 卷积层：0.1 ~ 0.3
    /// - 输入层后：0.1 ~ 0.2
    fn dropout(&self, p: f32) -> Result<Var, GraphError>;
}

impl VarRegularizationOps for Var {
    fn dropout(&self, p: f32) -> Result<Var, GraphError> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let graph = self.graph();
        // 使用时间戳作为随机种子
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let node = graph
            .borrow_mut()
            .create_dropout_node(Rc::clone(self.node()), p, seed, None)?;
        Ok(Self::new_with_rc_graph(node, &graph))
    }
}
