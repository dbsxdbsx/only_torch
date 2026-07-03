//! Tensor 共享存储（Arc/CoW）契约守门测试
//!
//! 锁死三条不变量：
//! 1. `clone()` 是浅拷贝——共享同一底层缓冲（O(1)，不复制数据）；
//! 2. 共享后的任一方发生可变访问时自动写时物化（CoW）——**值语义与深拷贝逐 bit 等价**，
//!    改一方不影响另一方；
//! 3. 独占（refcount == 1）时可变访问**不**触发重分配——热路径（如优化器就地更新参数）
//!    不会因 CoW 机制付隐藏拷贝。

use crate::tensor::Tensor;

/// 取张量底层缓冲的首元素地址（用于判定是否共享同一存储）
fn buf_ptr(t: &Tensor) -> *const f32 {
    t.data_as_slice().as_ptr()
}

#[test]
fn test_clone_shares_storage() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = a.clone();
    // 浅拷贝：底层缓冲同一块内存
    assert_eq!(buf_ptr(&a), buf_ptr(&b), "clone 后应共享同一底层缓冲");
    // source_id 语义保持：clone 同 ID（同一份数据）
    assert_eq!(a.source_id(), b.source_id());
}

#[test]
fn test_write_on_clone_materializes_and_preserves_value_semantics() {
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let mut b = a.clone();
    // 写 b：触发 CoW 物化，b 分家
    b[[0, 0]] = 42.0;
    assert_ne!(buf_ptr(&a), buf_ptr(&b), "写入共享张量后应物化出私有副本");
    // 值语义与深拷贝完全一致：a 不受影响
    assert_eq!(a[[0, 0]], 1.0);
    assert_eq!(b[[0, 0]], 42.0);
}

#[test]
fn test_write_on_original_does_not_affect_clone() {
    let mut a = Tensor::new(&[1.0, 2.0], &[2]);
    let b = a.clone();
    a += 10.0; // 就地运算走 DataMut → ensure_unique → CoW
    assert_eq!(a.data_as_slice(), &[11.0, 12.0]);
    assert_eq!(
        b.data_as_slice(),
        &[1.0, 2.0],
        "clone 持有者不应观察到原张量的修改"
    );
}

#[test]
fn test_unique_inplace_write_does_not_reallocate() {
    // 独占张量的就地写不应触发任何重分配（守护优化器参数更新等热路径）
    let mut a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let ptr_before = buf_ptr(&a);
    a *= 2.0;
    a += 1.0;
    assert_eq!(
        buf_ptr(&a),
        ptr_before,
        "独占（refcount==1）时就地写不应重分配缓冲"
    );
    assert_eq!(a.data_as_slice(), &[3.0, 5.0, 7.0]);
}

#[test]
fn test_shared_then_dropped_writes_in_place() {
    // 共享方全部释放后，剩余唯一持有者的就地写恢复零拷贝
    let mut a = Tensor::new(&[1.0, 2.0], &[2]);
    let ptr_orig = buf_ptr(&a);
    {
        let _b = a.clone();
    } // b 释放，refcount 回到 1
    a += 1.0;
    assert_eq!(buf_ptr(&a), ptr_orig, "共享方释放后就地写应复用原缓冲");
    assert_eq!(a.data_as_slice(), &[2.0, 3.0]);
}

#[test]
fn test_clone_of_noncontiguous_view_shares_storage() {
    // 非连续布局（permute 产物）clone 同样是浅拷贝
    let a = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).permute(&[1, 0]);
    let b = a.clone();
    assert!(!a.is_contiguous());
    // 非连续不能走 data_as_slice，比较逻辑值 + 写时值语义
    let mut c = b.clone();
    c[[0, 0]] = 99.0;
    assert_eq!(a[[0, 0]], 1.0, "非连续共享张量写时同样不应影响其他持有者");
    assert_eq!(c[[0, 0]], 99.0);
}
