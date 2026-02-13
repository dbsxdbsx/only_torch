/*
 * Tensor source_id 单元测试
 *
 * 验证数据源身份追踪机制的核心行为：
 * - 新 Tensor 获得唯一 ID
 * - clone 保持同一 ID
 * - 运算产生新 ID
 */

use crate::tensor::Tensor;

#[test]
fn test_new_tensors_have_unique_ids() {
    let a = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let b = Tensor::new(&[3.0, 4.0], &[1, 2]);
    assert_ne!(a.source_id(), b.source_id(), "不同 Tensor 应有不同 source_id");
}

#[test]
fn test_clone_preserves_source_id() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = a.clone();
    assert_eq!(
        a.source_id(),
        b.source_id(),
        "clone 应保持相同 source_id"
    );
}

#[test]
fn test_operations_produce_new_ids() {
    let a = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let b = Tensor::new(&[3.0, 4.0], &[1, 2]);
    let c = &a + &b;
    assert_ne!(c.source_id(), a.source_id(), "加法结果应有新 source_id");
    assert_ne!(c.source_id(), b.source_id(), "加法结果应有新 source_id");

    let d = &a * &b;
    assert_ne!(d.source_id(), a.source_id(), "乘法结果应有新 source_id");

    let e = &a - &b;
    assert_ne!(e.source_id(), a.source_id(), "减法结果应有新 source_id");

    let f = -&a;
    assert_ne!(f.source_id(), a.source_id(), "取反结果应有新 source_id");
}

#[test]
fn test_constructors_have_unique_ids() {
    let a = Tensor::zeros(&[2, 3]);
    let b = Tensor::ones(&[2, 3]);
    let c = Tensor::full(0.5, &[2, 3]);
    assert_ne!(a.source_id(), b.source_id());
    assert_ne!(b.source_id(), c.source_id());
    assert_ne!(a.source_id(), c.source_id());
}

#[test]
fn test_clone_chain_preserves_id() {
    // 模拟 SAC 场景：同一个 Tensor 传入多个 input_named
    let obs_batch = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let id = obs_batch.source_id();

    // set_value 内部会 clone，验证 clone 链的 ID 一致
    let copy1 = obs_batch.clone();
    let copy2 = obs_batch.clone();
    let copy3 = obs_batch.clone();
    assert_eq!(copy1.source_id(), id);
    assert_eq!(copy2.source_id(), id);
    assert_eq!(copy3.source_id(), id);
}
