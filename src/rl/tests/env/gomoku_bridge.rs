//! GymEnv 规划桥接测试（Gomoku 环境）
//!
//! 验证 Rust → Python Board 的 snapshot/restore/legal_mask 往返正确性。
//! 需要 Python + gymnasium + gym_env 包。

use pyo3::Python;
use serial_test::serial;

use crate::rl::GymEnv;

#[test]
#[serial]
fn test_gomoku_bridge_legal_mask() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Gomoku-selfplay-v0");
        env.reset(Some(42));

        let mask = env.legal_mask();
        assert_eq!(mask.len(), 81, "9×9 棋盘 legal_mask 长度应为 81");
        assert!(mask.iter().all(|&b| b), "初始棋盘所有位置应合法");

        // 下一步后掩码应少一个
        env.board_step(40); // 中心
        let mask2 = env.legal_mask();
        assert_eq!(mask2.len(), 81);
        assert!(!mask2[40], "已落子位置应不合法");
        assert_eq!(
            mask2.iter().filter(|&&b| b).count(),
            80,
            "落子后应有 80 个合法位置"
        );

        env.close();
    });
}

#[test]
#[serial]
fn test_gomoku_bridge_snapshot_restore() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Gomoku-selfplay-v0");
        env.reset(Some(42));

        // 保存空盘快照
        let snap = env.snapshot();

        // 下两步
        env.board_step(0);
        env.board_step(1);
        assert_eq!(env.current_player(), 0, "两步后应回到黑方");
        let mask_after = env.legal_mask();
        assert!(!mask_after[0]);
        assert!(!mask_after[1]);

        // 恢复
        env.restore(&snap);
        let mask_restored = env.legal_mask();
        assert!(mask_restored.iter().all(|&b| b), "恢复后所有位置应合法");
        assert_eq!(env.current_player(), 0, "恢复后应为黑方");

        env.close();
    });
}

#[test]
#[serial]
fn test_gomoku_bridge_current_player_alternates() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Gomoku-selfplay-v0");
        env.reset(Some(42));

        assert_eq!(env.current_player(), 0, "初始为黑方");
        env.board_step(40);
        assert_eq!(env.current_player(), 1, "黑落子后为白方");
        env.board_step(41);
        assert_eq!(env.current_player(), 0, "白落子后回到黑方");

        env.close();
    });
}

#[test]
#[serial]
fn test_gomoku_bridge_is_terminal() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Gomoku-selfplay-v0");
        env.reset(Some(42));

        assert!(!env.is_terminal(), "初始不应终局");

        // 构造黑方五连珠：(0,0) (0,1) (0,2) (0,3) (0,4)
        // 黑方下 0,2,4,6,8 位置（行0列0-4），白方下 9,10,11,12
        env.board_step(0); // 黑 (0,0)
        env.board_step(9); // 白 (1,0)
        env.board_step(1); // 黑 (0,1)
        env.board_step(10); // 白 (1,1)
        env.board_step(2); // 黑 (0,2)
        env.board_step(11); // 白 (1,2)
        env.board_step(3); // 黑 (0,3)
        env.board_step(12); // 白 (1,3)
        let (reward, terminal) = env.board_step(4); // 黑 (0,4) → 五连珠

        assert!(terminal, "黑方五连珠应终局");
        assert_eq!(reward, 1.0, "当前方（黑）胜奖励应为 1.0");
        assert!(env.is_terminal());

        env.close();
    });
}

#[test]
#[serial]
fn test_gomoku_bridge_observation_flat() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Gomoku-selfplay-v0");
        env.reset(Some(42));

        let obs = env.board_observation_flat();
        assert_eq!(obs.len(), 3 * 9 * 9, "展平观察长度应为 243");
        // 初始棋盘：通道 2（空位）全 1，通道 0、1 全 0
        let channel_size = 9 * 9;
        let ch0_sum: f32 = obs[..channel_size].iter().sum();
        let ch1_sum: f32 = obs[channel_size..2 * channel_size].iter().sum();
        let ch2_sum: f32 = obs[2 * channel_size..].iter().sum();
        assert_eq!(ch0_sum, 0.0, "初始黑子通道应全零");
        assert_eq!(ch1_sum, 0.0, "初始白子通道应全零");
        assert_eq!(ch2_sum, 81.0, "初始空位通道应全 1");

        env.close();
    });
}
