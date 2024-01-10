use crate::assert_panic;

#[test]
fn test_assert_panic_macro() {
    assert_panic!(panic!("test panic"));
    assert_panic!(panic!("custom test panic msg"), "custom test panic msg");
}
