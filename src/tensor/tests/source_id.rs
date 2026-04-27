/*
 * Tensor source_id 单元测试
 *
 * 核心语义：source_id 追踪的是**变量身份**（创建来源），不是数据内容。
 * - 新创建 → 新 ID
 * - clone → 保持 ID（同一来源的副本）
 * - 运算 → 新 ID（新的数据来源）
 * - 原地修改 → ID 不变（变量身份不变，只是内容更新）
 */

use crate::tensor::Tensor;

#[test]
fn test_new_tensors_have_unique_ids() {
    let a = Tensor::new(&[1.0, 2.0], &[1, 2]);
    let b = Tensor::new(&[3.0, 4.0], &[1, 2]);
    assert_ne!(
        a.source_id(),
        b.source_id(),
        "不同 Tensor 应有不同 source_id"
    );
}

#[test]
fn test_clone_preserves_source_id() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = a.clone();
    assert_eq!(a.source_id(), b.source_id(), "clone 应保持相同 source_id");
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

// === 原地修改相关 ===

#[test]
fn test_in_place_mutation_preserves_id() {
    // 原地修改数据不改变变量身份
    let mut a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let id_before = a.source_id();
    a += 10.0; // in-place 加标量
    assert_eq!(a.source_id(), id_before, "+= 不应改变 source_id");

    let b = Tensor::ones(&[2, 2]);
    a += &b; // in-place 加张量
    assert_eq!(a.source_id(), id_before, "+= Tensor 不应改变 source_id");

    a -= 1.0;
    assert_eq!(a.source_id(), id_before, "-= 不应改变 source_id");

    a *= 2.0;
    assert_eq!(a.source_id(), id_before, "*= 不应改变 source_id");

    a /= 2.0;
    assert_eq!(a.source_id(), id_before, "/= 不应改变 source_id");
}

#[test]
fn test_mutate_then_clone_preserves_id() {
    // 核心 corner case：修改数据后 clone，两者仍视为同源
    let mut a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let id = a.source_id();
    a += 100.0; // 数据已变
    let b = a.clone(); // clone 修改后的版本
    assert_eq!(
        a.source_id(),
        b.source_id(),
        "修改数据后 clone 应保持同一 source_id"
    );
    assert_eq!(a.source_id(), id, "修改不应改变原始 source_id");
}

// === 同值不同源 ===

#[test]
fn test_same_content_different_source() {
    // 内容完全相同但独立创建 → 不同源
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    assert_eq!(a, b, "内容应相同");
    assert_ne!(
        a.source_id(),
        b.source_id(),
        "内容相同但独立创建的 Tensor 应有不同 source_id"
    );
}
