"""
Tensor Add Broadcast Reference Test

Generate test cases for various shape combinations, for Rust unit test reference.
"""

import numpy as np

def test_broadcast_add(shape_a, shape_b):
    """
    Test if two shapes can be broadcast-added.
    
    Returns: (success, result_shape or error, data)
    """
    try:
        # 使用简单的递增数据
        size_a = int(np.prod(shape_a)) if shape_a else 1
        size_b = int(np.prod(shape_b)) if shape_b else 1
        
        a = np.arange(1, size_a + 1, dtype=np.float32).reshape(shape_a) if shape_a else np.array(1.0, dtype=np.float32)
        b = np.arange(1, size_b + 1, dtype=np.float32).reshape(shape_b) * 10 if shape_b else np.array(10.0, dtype=np.float32)
        
        result = a + b
        return (True, tuple(result.shape), a.flatten().tolist(), b.flatten().tolist(), result.flatten().tolist())
    except ValueError as e:
        return (False, str(e), None, None, None)


def main():
    print("=" * 70)
    print("NumPy 张量加法广播测试参考")
    print("=" * 70)
    
    # 测试用例分类
    test_cases = {
        "1. 相同形状（始终成功）": [
            ((), ()),           # 标量
            ((3,), (3,)),       # 向量
            ((2, 3), (2, 3)),   # 矩阵
            ((2, 3, 4), (2, 3, 4)),  # 3D
        ],
        
        "2. 标量与任意形状（始终成功）": [
            ((), (3,)),
            ((3,), ()),
            ((), (2, 3)),
            ((2, 3), ()),
            ((), (2, 3, 4)),
            ((2, 3, 4), ()),
        ],
        
        "3. Linear 层典型场景 [batch, out] + [1, out] 或 [out]": [
            ((3, 4), (1, 4)),   # batch=3, out=4
            ((3, 4), (4,)),     # 省略 batch 维度
            ((32, 128), (1, 128)),
            ((32, 128), (128,)),
        ],
        
        "4. Conv2d 典型场景 [batch, C, H, W] + [1, C, 1, 1] 或 [C, 1, 1]": [
            ((2, 3, 4, 4), (1, 3, 1, 1)),  # 标准 bias 广播
            ((2, 3, 4, 4), (3, 1, 1)),     # 省略 batch 维度
        ],
        
        "5. 不同维度数的合法广播": [
            ((3, 4), (4,)),     # 2D + 1D
            ((2, 3, 4), (4,)),  # 3D + 1D
            ((2, 3, 4), (3, 4)),  # 3D + 2D
            ((2, 3, 4), (1, 4)),  # 3D + 2D (with 1)
        ],
        
        "6. 维度为 1 的广播（合法）": [
            ((3, 1), (1, 4)),   # 结果 (3, 4)
            ((1, 4), (3, 1)),   # 结果 (3, 4)
            ((3, 1, 4), (1, 5, 1)),  # 结果 (3, 5, 4)
        ],
        
        "7. 不兼容的形状（应该失败）": [
            ((3,), (4,)),       # 维度不匹配
            ((2, 3), (3, 2)),   # 维度不匹配
            ((2, 3), (4,)),     # 最后一维不匹配
            ((2, 3, 4), (2, 4)),  # 中间维度不匹配
        ],
    }
    
    # 运行测试并输出结果
    all_results = []
    
    for category, cases in test_cases.items():
        print(f"\n{category}")
        print("-" * 60)
        
        for shape_a, shape_b in cases:
            success, result_shape, data_a, data_b, data_result = test_broadcast_add(shape_a, shape_b)
            
            if success:
                print(f"  OK: {list(shape_a)} + {list(shape_b)} -> {list(result_shape)}")
                all_results.append({
                    "shape_a": list(shape_a),
                    "shape_b": list(shape_b),
                    "result_shape": list(result_shape),
                    "data_a": data_a,
                    "data_b": data_b,
                    "data_result": data_result,
                    "success": True,
                })
            else:
                print(f"  FAIL: {list(shape_a)} + {list(shape_b)}")
                all_results.append({
                    "shape_a": list(shape_a),
                    "shape_b": list(shape_b),
                    "success": False,
                })
    
    # Output Rust test code template
    print("\n")
    print("=" * 70)
    print("Rust Test Code Template (Success Cases)")
    print("=" * 70)
    
    print("""
/// Broadcast add test cases (based on NumPy reference)
#[test]
fn test_add_broadcast() {
    // Format: (shape_a, data_a, shape_b, data_b, result_shape, expected_result)
    let test_cases: &[(&[usize], &[f32], &[usize], &[f32], &[usize], &[f32])] = &[""")
    
    for r in all_results:
        if r["success"]:
            shape_a = r["shape_a"]
            shape_b = r["shape_b"]
            result_shape = r["result_shape"]
            data_a = r["data_a"]
            data_b = r["data_b"]
            data_result = r["data_result"]
            
            # 格式化数据（保留一位小数）
            data_a_str = ", ".join(f"{x:.1f}" for x in data_a)
            data_b_str = ", ".join(f"{x:.1f}" for x in data_b)
            data_result_str = ", ".join(f"{x:.1f}" for x in data_result)
            
            print(f"        // {shape_a} + {shape_b} -> {result_shape}")
            print(f"        (&{shape_a}, &[{data_a_str}], &{shape_b}, &[{data_b_str}], &{result_shape}, &[{data_result_str}]),")
    
    print("""    ];
    
    for (shape_a, data_a, shape_b, data_b, expected_shape, expected_data) in test_cases {
        let a = Tensor::new(data_a, shape_a);
        let b = Tensor::new(data_b, shape_b);
        let result = a + b;
        
        assert_eq!(result.shape(), *expected_shape, 
            "Shape mismatch: {:?} + {:?} expected {:?}, got {:?}", 
            shape_a, shape_b, expected_shape, result.shape());
        
        let result_data: Vec<f32> = result.data().iter().cloned().collect();
        assert_eq!(result_data, expected_data.to_vec(),
            "Data mismatch: {:?} + {:?}", shape_a, shape_b);
    }
}""")

    print("\n")
    print("=" * 70)
    print("Rust Test Code Template (Failure Cases)")
    print("=" * 70)
    
    for r in all_results:
        if not r["success"]:
            shape_a = r["shape_a"]
            shape_b = r["shape_b"]
            print(f"""
#[test]
#[should_panic]
fn test_add_broadcast_fail_{shape_a}_{shape_b}() {{
    let a = Tensor::new(&[1.0; {max(1, int(np.prod(shape_a)))}], &{shape_a});
    let b = Tensor::new(&[1.0; {max(1, int(np.prod(shape_b)))}], &{shape_b});
    let _ = a + b;  // 应该 panic
}}""".replace("()", "[]"))


if __name__ == "__main__":
    main()
