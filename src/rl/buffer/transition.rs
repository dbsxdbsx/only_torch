//! 单步交互数据类型

/// `ReplayBuffer` 存储元素的约束。
///
/// `T` 必须是**纯 owned 数据**，不得持 `PyObject` / 借用 / 短生命周期引用。
/// CPU-only 单线程无跨线程需求，故不要求 `Send`。
pub trait BufferItem: Clone + 'static {}

/// 单步交互（off-policy buffer 的最小单位）。
///
/// # action 编码约定
/// - **离散**（Discrete(n)）：`vec![idx as f32]`，读取时 `action[0] as usize`
/// - **连续**（Box(d,)）：长度 = d 的 `Vec<f32>`，按 Gymnasium 顺序
/// - **混合 Tuple(Discrete(n), Box(d,))**：`[idx as f32, c_0, c_1, …, c_{d-1}]`
///   （离散在前，连续在后）
///
/// 同一示例内必须保持编码一致；解码集中在网络入口或 helper 内，禁止散布到训练循环。
///
/// # 终止语义（镜像 Gymnasium，勿合并成单一 done）
/// - `terminated`：MDP 真终止（杆倒了 / 到目标）→ **不** bootstrap
/// - `truncated`：外部截断（时间 / 步数上限）→ **仍** bootstrap
/// - TD target：`r + γ·(1 - terminated as f32)·V(next)`
/// - 用户从 `env.step()` 直接搬入，不要自己做布尔合并
#[derive(Debug, Clone)]
pub struct Transition {
    pub obs: Vec<f32>,
    pub action: Vec<f32>,
    pub reward: f32,
    pub next_obs: Vec<f32>,
    pub terminated: bool,
    pub truncated: bool,
}

impl Transition {
    /// 回合是否结束（统计/收集循环用），与 bootstrap 无关。
    pub fn is_episode_end(&self) -> bool {
        self.terminated || self.truncated
    }
}

impl BufferItem for Transition {}
