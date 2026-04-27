//! 9×10 棋盘对齐 + FEN 序列化
//!
//! 中国象棋棋盘 9 列 × 10 行。**标准 FEN 约定红方在 row 9 底,跟原图视觉朝向无关**——
//! FEN 是逻辑棋局表示,字符串本身**无法表达**原图视觉上红方在上还是在下。
//! 因此本模块把"视觉朝向"作为**独立元信息**输出(`detect_red_on_top` 返回 bool),
//! 调用方可以同时拿到\[标准 FEN + 原图视觉朝向\]两份信息,各取所需。
//!
//! 本模块负责：
//! 1. 把 letterbox 空间下的 bbox 中心反映射到原图坐标
//! 2. 自动锁定棋盘 ROI(优先用棋子检测包络,fallback 用第 15 个 board 类的 bbox)
//! 3. 按棋盘 ROI 把每个棋子归并到对应格点(同格多检出取最高 conf)
//! 4. **视觉朝向独立检测**:看红帅(r_jiang)在 ROI 上半还是下半;在上半时
//!    `rotate_grid_180` 把整盘转回标准方向,然后序列化的 FEN 仍然遵循"红方在底"约定
//! 5. 序列化为 FEN 字符串(中国象棋方言)
//!
//! ## 类别约定(VinXiangQi v1.4.0 官方源码 YoloXiangQiModel.cs)
//!
//! 15 类(0-indexed,1-indexed 是 C# 源码里的 Id 字段):
//! ```
//!  0: b_ma   (黑馬,FEN n)     8: r_ma     (红馬,FEN N)
//!  1: b_xiang(黑象,FEN b)     9: r_shi    (红仕,FEN A)
//!  2: b_shi  (黑士,FEN a)    10: r_jiang  (红帥,FEN K)
//!  3: b_jiang(黑將,FEN k)    11: r_xiang  (红相,FEN B)
//!  4: b_che  (黑車,FEN r)    12: r_pao    (红炮,FEN C)
//!  5: b_pao  (黑炮,FEN c)    13: r_bing   (红兵,FEN P)
//!  6: b_bing (黑卒,FEN p)    14: board    (整个棋盘 bbox,用于 ROI 自动锁定)
//!  7: r_che  (红車,FEN R)
//! ```
//!
//! ## FEN 表示
//!
//! 行从上到下、列从左到右;空格用数字累计;行用 "/" 分隔。
//! 标准 FEN 约定红方在 row 9(底)、黑方在 row 0(顶)。
//!
//! 例:初始局面(仅含子方):
//!     `rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR`

use crate::letterbox::LetterboxResult;
use crate::yolo_decode::Detection;

pub const BOARD_COLS: usize = 9;
pub const BOARD_ROWS: usize = 10;
/// 实际棋子类别数(不含 board)。模型最后一维 = 5 + (NUM_CLASSES + 1) = 20。
pub const NUM_CLASSES: usize = 14;
/// 第 15 类(class_id = 14)是 board 整体 bbox,用于自动锁定 ROI。
pub const BOARD_CLASS_ID: usize = 14;

/// 棋盘 ROI 与类别字典配置
pub struct BoardConfig {
    /// 棋盘 ROI 在**原始截图**坐标系下的 (x0, y0, x1, y1)
    /// 仅含 9×10 格点构成的矩形(不含外围装饰边)
    pub roi: (u32, u32, u32, u32),
    /// 类别索引 → FEN 字符(长度必须等于 NUM_CLASSES)
    pub class_to_fen: [char; NUM_CLASSES],
}

impl BoardConfig {
    /// VinXiangQi 默认类别字典(按官方源码 YoloXiangQiModel.cs)
    ///
    /// 索引 0..6 是黑方(小写),7..13 是红方(大写)
    pub fn default_class_to_fen() -> [char; NUM_CLASSES] {
        [
            'n', 'b', 'a', 'k', 'r', 'c', 'p', 'R', 'N', 'A', 'K', 'B', 'C', 'P',
        ]
    }
}

/// 从 detections 自动估计棋盘 ROI(返回 9×10 个**格点**构成的矩形):
/// - x0, y0 = (0,0) 格点的中心坐标
/// - x1, y1 = (BOARD_ROWS-1, BOARD_COLS-1) 格点的中心坐标
///
/// 算法:
/// 1. 优先用棋子检测的中心坐标包络(很鲁棒,只要 ≥4 个棋子分布在四角附近就行)
/// 2. 棋子太少(<4) 时退回 board 类(class 14)的 bbox 中心区域
/// 3. 都不行返回 None
///
/// 注意:VinXiangQi 的 "board" 类 bbox 在 only_torch 推理路径下数值偏小(框架内
/// 部数值漂移导致),不能直接当 ROI 用,只能作 fallback。
pub fn auto_detect_board_roi(
    dets: &[Detection],
    letterbox: &LetterboxResult,
) -> Option<(u32, u32, u32, u32)> {
    // Path 1: 棋子中心坐标包络(优先)
    let pieces: Vec<(f32, f32)> = dets
        .iter()
        .filter(|d| d.class_id < NUM_CLASSES)
        .map(|d| {
            let (cx_lb, cy_lb) = d.center();
            letterbox.to_origin(cx_lb, cy_lb)
        })
        .collect();
    if pieces.len() >= 4 {
        let mut min_cx = f32::INFINITY;
        let mut max_cx = f32::NEG_INFINITY;
        let mut min_cy = f32::INFINITY;
        let mut max_cy = f32::NEG_INFINITY;
        for (cx, cy) in &pieces {
            min_cx = min_cx.min(*cx);
            max_cx = max_cx.max(*cx);
            min_cy = min_cy.min(*cy);
            max_cy = max_cy.max(*cy);
        }
        return Some((
            min_cx.max(0.0) as u32,
            min_cy.max(0.0) as u32,
            max_cx.max(0.0) as u32,
            max_cy.max(0.0) as u32,
        ));
    }

    // Path 2(fallback): board 类(class 14)bbox 中心区域
    // VinXiangQi 训练时给整个棋盘外接矩形标了 board 类,bbox 比格点矩形稍大
    // (含外围装饰线),所以这里取 bbox 中心 + 适当内缩
    let board_det = dets
        .iter()
        .filter(|d| d.class_id == BOARD_CLASS_ID)
        .max_by(|a, b| {
            a.conf
                .partial_cmp(&b.conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some(d) = board_det {
        let (x0_lb, y0_lb, x1_lb, y1_lb) = (d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]);
        let (x0, y0) = letterbox.to_origin(x0_lb, y0_lb);
        let (x1, y1) = letterbox.to_origin(x1_lb, y1_lb);
        // 内缩 bbox 的 5% 作格点矩形(经验估计:bbox 比格点矩形稍大)
        let dx = (x1 - x0) * 0.05;
        let dy = (y1 - y0) * 0.05;
        return Some((
            (x0 + dx).max(0.0) as u32,
            (y0 + dy).max(0.0) as u32,
            (x1 - dx).max(0.0) as u32,
            (y1 - dy).max(0.0) as u32,
        ));
    }

    None
}

/// 检测视觉朝向:返回 `true` 表示原图里红方在棋盘**上方**(=黑方在下),需要旋转 180°
///
/// 启发式:看红帅(class_id = 10, r_jiang)的 cy 在 ROI 哪一半。
/// 在上半 → 红方在上(返回 true);在下半 → 标准方向(返回 false,不旋转)。
///
/// **这是独立元信息**,不影响 FEN 字符串内容——FEN 永远输出"红方在 row 9 底"
/// 的标准化形式。本函数的返回值供调用方报告"原图视觉朝向"使用。
///
/// 没检出红帅时返回 `false`(默认按标准方向处理),实测内置两张 sample 截图
/// 红帅都能稳定检出,fallback 暂未实现。
pub fn detect_red_on_top(
    dets: &[Detection],
    letterbox: &LetterboxResult,
    roi: (u32, u32, u32, u32),
) -> bool {
    let red_jiang = dets
        .iter()
        .filter(|d| d.class_id == 10) // r_jiang
        .max_by(|a, b| {
            a.conf
                .partial_cmp(&b.conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some(d) = red_jiang {
        let (_, cy_lb) = d.center();
        let (_, cy) = letterbox.to_origin(0.0, cy_lb);
        let mid_y = (roi.1 + roi.3) as f32 / 2.0;
        return cy < mid_y;
    }
    false
}

/// 9×10 网格（[row][col]），每格 `Some(class_id)` 或 `None`
pub type Grid = [[Option<usize>; BOARD_COLS]; BOARD_ROWS];

/// 把检测列表对齐到 9×10 网格
///
/// 算法：
/// 1. 把 detection 中心从 letterbox 空间反映射到原图坐标
/// 2. 用棋盘 ROI 计算 cell_w / cell_h
/// 3. ROI 包含 9×10 个**格点**(intersection),格点(0,0)在 ROI 左上角,
///    格点(row, col) 的中心 = (x0 + col * (x1-x0)/8, y0 + row * (y1-y0)/9)
///    `col = round((cx - x0) / cell_w)`、`row = round((cy - y0) / cell_h)`
/// 4. 同格多检出：保留最高 conf
/// 5. 跳过 board 类(`class_id == BOARD_CLASS_ID`,只用于 ROI 检测)
///
/// # 注意
/// - 落在棋盘外的 detection 被丢弃
/// - 行/列约束在 [0, 10) / [0, 9) 之间
pub fn align_to_grid(dets: &[Detection], letterbox: &LetterboxResult, cfg: &BoardConfig) -> Grid {
    let (x0, y0, x1, y1) = cfg.roi;
    // 注意:9 列对应 8 个间隔,10 行对应 9 个间隔(格点 vs 格子的区别)
    let cell_w = (x1 - x0) as f32 / (BOARD_COLS - 1) as f32;
    let cell_h = (y1 - y0) as f32 / (BOARD_ROWS - 1) as f32;

    let mut grid: Grid = [[None; BOARD_COLS]; BOARD_ROWS];
    let mut grid_conf = [[0f32; BOARD_COLS]; BOARD_ROWS];

    for d in dets {
        // 跳过 board 整体类(只用于 ROI 检测,不进 grid)
        if d.class_id >= NUM_CLASSES {
            continue;
        }

        let (cx_lb, cy_lb) = d.center();
        let (cx, cy) = letterbox.to_origin(cx_lb, cy_lb);

        // 给 ROI 半个 cell 容差(格点本身可能恰好在 ROI 边缘)
        let margin_x = cell_w * 0.5;
        let margin_y = cell_h * 0.5;
        if cx < x0 as f32 - margin_x
            || cx > x1 as f32 + margin_x
            || cy < y0 as f32 - margin_y
            || cy > y1 as f32 + margin_y
        {
            continue;
        }

        let col_f = (cx - x0 as f32) / cell_w;
        let row_f = (cy - y0 as f32) / cell_h;
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

/// 把 grid 旋转 180°(原 (row, col) 移到 (BOARD_ROWS-1-row, BOARD_COLS-1-col))
///
/// 用于"红方在上"的截图:旋转后红方回到 row 9(底),符合标准 FEN 约定。
pub fn rotate_grid_180(grid: &Grid) -> Grid {
    let mut out: Grid = [[None; BOARD_COLS]; BOARD_ROWS];
    for r in 0..BOARD_ROWS {
        for c in 0..BOARD_COLS {
            out[BOARD_ROWS - 1 - r][BOARD_COLS - 1 - c] = grid[r][c];
        }
    }
    out
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
