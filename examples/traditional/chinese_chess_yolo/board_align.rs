//! 9×10 棋盘对齐 + FEN 序列化
//!
//! 中国象棋棋盘 9 列 × 10 行（从顶部红方起算）。本模块负责：
//! 1. 把 letterbox 空间下的 bbox 中心反映射到原图坐标
//! 2. 按棋盘 ROI 把每个棋子归并到对应格点（同格多检出取最高 conf）
//! 3. 序列化为 FEN 字符串（中国象棋方言）
//!
//! ## 类别约定（VinXiangQi 模型）
//!
//! 14 类按红/黑各 7 种排列：
//! ```
//!  0..6  红方: R(车) N(马) C(炮) K(将) A(士) B(相) P(兵)
//!  7..13 黑方: r(车) n(马) c(炮) k(将) a(士) b(象) p(卒)
//! ```
//!
//! ## FEN 表示
//!
//! 行从上到下、列从左到右；空格用数字累计；行用 "/" 分隔。
//! 例：初始局面（仅含子方）：
//!     `rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR`

use crate::yolo_decode::Detection;
use crate::letterbox::LetterboxResult;

pub const BOARD_COLS: usize = 9;
pub const BOARD_ROWS: usize = 10;
pub const NUM_CLASSES: usize = 14;

/// 棋盘 ROI 与类别字典配置
pub struct BoardConfig {
    /// 棋盘 ROI 在**原始截图**坐标系下的 (x0, y0, x1, y1)
    /// 仅含 9×10 格点构成的矩形（不含外围装饰边）
    pub roi: (u32, u32, u32, u32),
    /// 类别索引 → FEN 字符（长度必须等于 NUM_CLASSES）
    pub class_to_fen: [char; NUM_CLASSES],
}

impl BoardConfig {
    /// VinXiangQi 默认类别字典（按发布者约定，可能因模型版本不同而调整）
    pub fn default_class_to_fen() -> [char; NUM_CLASSES] {
        // 红方在前，索引 0..6 对应 RNCKABP；黑方在后，索引 7..13 对应小写
        ['R', 'N', 'C', 'K', 'A', 'B', 'P', 'r', 'n', 'c', 'k', 'a', 'b', 'p']
    }
}

/// 9×10 网格（[row][col]），每格 `Some(class_id)` 或 `None`
pub type Grid = [[Option<usize>; BOARD_COLS]; BOARD_ROWS];

/// 把检测列表对齐到 9×10 网格
///
/// 算法：
/// 1. 把 detection 中心从 letterbox 空间反映射到原图坐标
/// 2. 用棋盘 ROI 计算 cell_w / cell_h
/// 3. col = round((cx - x0) / cell_w - 0.5)、row = round((cy - y0) / cell_h - 0.5)
/// 4. 同格多检出：保留最高 conf
///
/// # 注意
/// - 落在棋盘外的 detection 被丢弃
/// - 行/列约束在 [0, 10) / [0, 9) 之间
pub fn align_to_grid(
    dets: &[Detection],
    letterbox: &LetterboxResult,
    cfg: &BoardConfig,
) -> Grid {
    let (x0, y0, x1, y1) = cfg.roi;
    let cell_w = (x1 - x0) as f32 / BOARD_COLS as f32;
    let cell_h = (y1 - y0) as f32 / BOARD_ROWS as f32;

    let mut grid: Grid = [[None; BOARD_COLS]; BOARD_ROWS];
    let mut grid_conf = [[0f32; BOARD_COLS]; BOARD_ROWS];

    for d in dets {
        let (cx_lb, cy_lb) = d.center();
        let (cx, cy) = letterbox.to_origin(cx_lb, cy_lb);

        if cx < x0 as f32 || cx > x1 as f32 || cy < y0 as f32 || cy > y1 as f32 {
            continue;
        }

        let col_f = (cx - x0 as f32) / cell_w - 0.5;
        let row_f = (cy - y0 as f32) / cell_h - 0.5;
        let col = col_f.round() as i32;
        let row = row_f.round() as i32;

        if col < 0 || col >= BOARD_COLS as i32 || row < 0 || row >= BOARD_ROWS as i32 {
            continue;
        }
        let col = col as usize;
        let row = row as usize;

        if d.conf > grid_conf[row][col] {
            grid_conf[row][col] = d.conf;
            grid[row][col] = Some(d.class_id);
        }
    }
    grid
}

/// 把 9×10 grid 序列化为 FEN 字符串
///
/// 标准格式：行从上到下，列从左到右；空格用数字累计；行间用 "/" 分隔。
pub fn to_fen(grid: &Grid, cfg: &BoardConfig) -> String {
    let mut s = String::new();
    for (row_idx, row) in grid.iter().enumerate() {
        let mut empty_run = 0u32;
        for cell in row.iter() {
            match cell {
                Some(class_id) => {
                    if empty_run > 0 {
                        s.push_str(&empty_run.to_string());
                        empty_run = 0;
                    }
                    if *class_id < NUM_CLASSES {
                        s.push(cfg.class_to_fen[*class_id]);
                    } else {
                        s.push('?');
                    }
                }
                None => empty_run += 1,
            }
        }
        if empty_run > 0 {
            s.push_str(&empty_run.to_string());
        }
        if row_idx + 1 < BOARD_ROWS {
            s.push('/');
        }
    }
    s
}

/// 统计 grid 中非空棋子数量
pub fn count_pieces(grid: &Grid) -> usize {
    grid.iter()
        .flat_map(|row| row.iter())
        .filter(|c| c.is_some())
        .count()
}
