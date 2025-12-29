use std::path::Path;

/// 错误断言宏 - 灵活粒度验证 Result 错误
///
/// # 用法
/// - `assert_err!(expr)` — 只验证是 Err
/// - `assert_err!(expr, Variant(literal))` — 验证错误类型 + 精确消息（String 变体）
/// - `assert_err!(expr, ShapeMismatch(exp, got, msg))` — 验证 ShapeMismatch（简洁语法）
/// - `assert_err!(expr, Pattern { .. })` — 验证错误类型
/// - `assert_err!(expr, Pattern { field, .. } if condition)` — 验证类型 + 条件
///
/// # 示例
/// ```ignore
/// // 只验证是错误
/// assert_err!(result);
///
/// // 验证错误类型 + 精确消息（简洁语法）
/// assert_err!(result, GraphError::InvalidOperation("Add节点至少需要2个父节点"));
///
/// // ShapeMismatch 简洁语法（按顺序：expected, got, message）
/// assert_err!(result, GraphError::ShapeMismatch([2, 2], [3, 2], "消息"));
///
/// // 验证错误类型（忽略所有字段）
/// assert_err!(result, GraphError::ShapeMismatch { .. });
///
/// // 验证类型 + 关键字段
/// assert_err!(result, GraphError::ShapeMismatch { expected, .. } if expected == &[2, 2]);
///
/// // 验证消息包含关键词
/// assert_err!(result, GraphError::InvalidOperation(msg) if msg.contains("cycle"));
/// ```
#[macro_export]
macro_rules! assert_err {
    // 只验证是 Err
    ($expr:expr) => {
        assert!($expr.is_err(), "预期 Err，实际得到 {:?}", $expr);
    };
    // 简洁语法：Variant(字符串字面量) - 精确匹配 String 内容
    ($expr:expr, $err_type:ident :: $variant:ident ( $expected:literal )) => {
        match &$expr {
            Err($err_type::$variant(actual)) => assert_eq!(
                actual, $expected,
                "错误消息不匹配：预期 `{}`，实际得到 `{}`",
                $expected, actual
            ),
            Err(e) => panic!(
                "错误类型不匹配：预期 `{}::{}`，实际得到 `{:?}`",
                stringify!($err_type), stringify!($variant), e
            ),
            Ok(v) => panic!(
                "预期 Err({}::{})，实际得到 Ok({:?})",
                stringify!($err_type), stringify!($variant), v
            ),
        }
    };
    // 简洁语法：ShapeMismatch(expected, got, message)
    ($expr:expr, $err_type:ident :: ShapeMismatch ( $exp:expr, $got:expr, $msg:expr )) => {
        match &$expr {
            Err($err_type::ShapeMismatch { expected, got, message }) => {
                assert_eq!(expected.as_slice(), &$exp, "expected 不匹配");
                assert_eq!(got.as_slice(), &$got, "got 不匹配");
                assert_eq!(message, $msg, "message 不匹配");
            }
            Err(e) => panic!(
                "错误类型不匹配：预期 `{}::ShapeMismatch`，实际得到 `{:?}`",
                stringify!($err_type), e
            ),
            Ok(v) => panic!(
                "预期 Err({}::ShapeMismatch)，实际得到 Ok({:?})",
                stringify!($err_type), v
            ),
        }
    };
    // 简洁语法：DimensionMismatch(expected, got)
    ($expr:expr, $err_type:ident :: DimensionMismatch ( $exp:expr, $got:expr )) => {
        match &$expr {
            Err($err_type::DimensionMismatch { expected, got }) => {
                assert_eq!(*expected, $exp, "expected 不匹配");
                assert_eq!(*got, $got, "got 不匹配");
            }
            Err(e) => panic!(
                "错误类型不匹配：预期 `{}::DimensionMismatch`，实际得到 `{:?}`",
                stringify!($err_type), e
            ),
            Ok(v) => panic!(
                "预期 Err({}::DimensionMismatch)，实际得到 Ok({:?})",
                stringify!($err_type), v
            ),
        }
    };
    // 简洁语法：NodeNotFound(id)
    ($expr:expr, $err_type:ident :: NodeNotFound ( $id:expr )) => {
        match &$expr {
            Err($err_type::NodeNotFound { id }) => {
                assert_eq!(*id, $id, "id 不匹配");
            }
            Err(e) => panic!(
                "错误类型不匹配：预期 `{}::NodeNotFound`，实际得到 `{:?}`",
                stringify!($err_type), e
            ),
            Ok(v) => panic!(
                "预期 Err({}::NodeNotFound)，实际得到 Ok({:?})",
                stringify!($err_type), v
            ),
        }
    };
    // 通用模式匹配（带 if guard 或复杂 pattern）
    ($expr:expr, $($pattern:tt)+) => {
        match &$expr {
            Err(e) => assert!(
                matches!(e, $($pattern)+),
                "错误类型不匹配：预期 `{}`，实际得到 `{:?}`",
                stringify!($($pattern)+),
                e
            ),
            Ok(v) => panic!(
                "预期 Err 匹配 `{}`，实际得到 Ok({:?})",
                stringify!($($pattern)+),
                v
            ),
        }
    };
}

#[macro_export]
macro_rules! assert_panic {
    ($expr:expr) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $expr)) {
            Ok(_) => panic!("表达式没有触发panic"),
            Err(_) => (),
        }
    };
    ($expr:expr, $expected_msg:expr) => {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $expr)) {
            Ok(_) => panic!("表达式没有触发panic"),
            Err(err) => {
                let expected_msg_str = $expected_msg.to_string();
                if let Some(msg) = err.downcast_ref::<&'static str>() {
                    assert_eq!(*msg, expected_msg_str, "panic消息与预期不符");
                } else if let Some(msg) = err.downcast_ref::<String>() {
                    assert_eq!(*msg, expected_msg_str, "panic消息与预期不符");
                } else {
                    panic!(
                        "未找到预期的panic消息，预期的panic消息为: {}",
                        expected_msg_str
                    );
                }
            }
        }
    };
}

pub fn get_file_size_in_byte(path: impl AsRef<Path>) -> u64 {
    let metadata = std::fs::metadata(path).unwrap();

    metadata.len()
}
