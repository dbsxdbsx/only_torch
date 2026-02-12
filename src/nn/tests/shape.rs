/*
 * DynamicShape 单元测试（动态维度形状系统）
 *
 * 测试形状创建、显示、兼容性判断、合并、具体化、广播等功能。
 */

use crate::nn::DynamicShape;

#[test]
fn test_dynamic_shape_creation() {
    // 固定形状
    let fixed = DynamicShape::fixed(&[32, 128]);
    assert_eq!(fixed.ndim(), 2);
    assert!(!fixed.has_dynamic_dims());
    assert_eq!(fixed.dim(0), Some(32));
    assert_eq!(fixed.dim(1), Some(128));

    // 动态 batch
    let dynamic_batch = DynamicShape::with_dynamic_batch(&[128]);
    assert_eq!(dynamic_batch.ndim(), 2);
    assert!(dynamic_batch.has_dynamic_dims());
    assert!(dynamic_batch.is_dynamic(0));
    assert!(!dynamic_batch.is_dynamic(1));
    assert_eq!(dynamic_batch.dim(0), None);
    assert_eq!(dynamic_batch.dim(1), Some(128));

    // 自定义
    let custom = DynamicShape::new(&[None, Some(10), None, Some(64)]);
    assert_eq!(custom.ndim(), 4);
    assert!(custom.is_dynamic(0));
    assert!(!custom.is_dynamic(1));
    assert!(custom.is_dynamic(2));
    assert!(!custom.is_dynamic(3));
}

#[test]
fn test_dynamic_shape_display() {
    assert_eq!(DynamicShape::fixed(&[32, 128]).to_string(), "[32, 128]");
    assert_eq!(
        DynamicShape::with_dynamic_batch(&[128]).to_string(),
        "[?, 128]"
    );
    assert_eq!(
        DynamicShape::new(&[None, Some(10), None]).to_string(),
        "[?, 10, ?]"
    );
}

#[test]
fn test_dynamic_shape_compatibility() {
    let dynamic = DynamicShape::new(&[None, Some(128)]);
    let fixed1 = DynamicShape::fixed(&[32, 128]);
    let fixed2 = DynamicShape::fixed(&[16, 128]);
    let fixed3 = DynamicShape::fixed(&[32, 64]);
    let fixed4 = DynamicShape::fixed(&[32, 128, 10]);

    // 动态与固定兼容
    assert!(dynamic.is_compatible(&fixed1));
    assert!(dynamic.is_compatible(&fixed2));

    // 固定维度不匹配
    assert!(!dynamic.is_compatible(&fixed3));

    // 维度数不同
    assert!(!dynamic.is_compatible(&fixed4));

    // 与张量形状兼容
    assert!(dynamic.is_compatible_with_tensor(&[32, 128]));
    assert!(dynamic.is_compatible_with_tensor(&[1, 128]));
    assert!(!dynamic.is_compatible_with_tensor(&[32, 64]));
}

#[test]
fn test_dynamic_shape_merge() {
    let a = DynamicShape::new(&[None, Some(128)]);
    let b = DynamicShape::fixed(&[32, 128]);

    let merged = a.merge(&b).unwrap();
    assert_eq!(merged.to_string(), "[32, 128]");

    // 不兼容的形状无法合并
    let c = DynamicShape::fixed(&[32, 64]);
    assert!(a.merge(&c).is_none());

    // 两个动态合并仍是动态
    let d = DynamicShape::new(&[None, Some(128)]);
    let merged2 = a.merge(&d).unwrap();
    assert_eq!(merged2.to_string(), "[?, 128]");
}

#[test]
fn test_dynamic_shape_concretize() {
    let shape = DynamicShape::new(&[None, Some(128)]);

    let concrete = shape.concretize(&[32, 128]).unwrap();
    assert_eq!(concrete.to_vec_fixed().unwrap(), vec![32, 128]);

    // 不兼容的形状无法具体化
    assert!(shape.concretize(&[32, 64]).is_none());
}

#[test]
fn test_feature_shape() {
    let shape = DynamicShape::fixed(&[32, 128, 64]);
    let features = shape.feature_shape();
    assert_eq!(features.to_string(), "[128, 64]");

    let dynamic = DynamicShape::with_dynamic_batch(&[128, 64]);
    let features2 = dynamic.feature_shape();
    assert_eq!(features2.to_string(), "[128, 64]");
}

#[test]
fn test_with_batch_dynamic() {
    let fixed = DynamicShape::fixed(&[32, 128]);
    let dynamic = fixed.with_batch_dynamic();
    assert_eq!(dynamic.to_string(), "[?, 128]");
}

#[test]
fn test_to_vec_fixed() {
    let fixed = DynamicShape::fixed(&[32, 128]);
    assert_eq!(fixed.to_vec_fixed(), Some(vec![32, 128]));

    let dynamic = DynamicShape::with_dynamic_batch(&[128]);
    assert_eq!(dynamic.to_vec_fixed(), None);
}

#[test]
fn test_broadcast_with() {
    // 动态与固定广播
    let a = DynamicShape::new(&[None, Some(128)]);
    let b = DynamicShape::fixed(&[1, 128]);
    let result = a.broadcast_with(&b);
    assert_eq!(result.to_string(), "[?, 128]");

    // 不同维度数的广播（右对齐）
    let c = DynamicShape::fixed(&[128]);
    let d = DynamicShape::fixed(&[32, 128]);
    let result2 = c.broadcast_with(&d);
    // 右对齐后 [?, 128] 和 [32, 128]
    // 左边扩展用 None 填充，结果是 [?, 128]
    assert_eq!(result2.ndim(), 2);

    // 两个固定形状广播
    let e = DynamicShape::fixed(&[1, 128]);
    let f = DynamicShape::fixed(&[32, 128]);
    let result3 = e.broadcast_with(&f);
    assert_eq!(result3.to_string(), "[32, 128]");

    // 动态维度传播
    let g = DynamicShape::new(&[None, Some(64)]);
    let h = DynamicShape::new(&[Some(32), None]);
    let result4 = g.broadcast_with(&h);
    // 任一动态则结果动态
    assert_eq!(result4.to_string(), "[?, ?]");
}
