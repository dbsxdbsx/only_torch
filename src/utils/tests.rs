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
