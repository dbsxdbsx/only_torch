"""
Tensor Div (/) Broadcast Reference Test

Test cases for element-wise division with broadcasting.

Note:
- Division is NOT commutative (a / b != b / a)
- Only Torch panics on division by zero (unlike NumPy which returns inf/nan)
- All test data avoids zero divisors

Reference script for Rust unit tests.
"""

import numpy as np


def test_broadcast_div(shape_a, shape_b):
    """
    Test if a / b succeeds with broadcasting.
    Uses data that avoids zero values in divisor.

    Returns: (success, result_shape, a_data, b_data, result_data) or (False, error)
    """
    try:
        size_a = int(np.prod(shape_a)) if shape_a else 1
        size_b = int(np.prod(shape_b)) if shape_b else 1

        # Use values starting from 1 to avoid zero divisors
        a = np.arange(1, size_a + 1, dtype=np.float32).reshape(shape_a) if shape_a else np.array(10.0, dtype=np.float32)
        b = np.arange(1, size_b + 1, dtype=np.float32).reshape(shape_b) if shape_b else np.array(2.0, dtype=np.float32)

        result = a / b
        return (True, tuple(result.shape), a.flatten().tolist(), b.flatten().tolist(), result.flatten().tolist())
    except ValueError as e:
        return (False, str(e), None, None, None)


def main():
    print("=" * 70)
    print("NumPy / Broadcast Reference Test")
    print("=" * 70)

    # Test cases: (shape_a, shape_b, description)
    test_cases = [
        # 1. Same shape
        ((3,), (3,), "same shape vector"),
        ((2, 3), (2, 3), "same shape matrix"),

        # 2. Scalar broadcast
        ((), (3,), "scalar / vector"),
        ((3,), (), "vector / scalar"),
        ((), (2, 3), "scalar / matrix"),
        ((2, 3), (), "matrix / scalar"),

        # 3. Lower dim broadcast to higher dim
        ((3, 4), (4,), "[batch,out] / [out] (normalize)"),
        ((3, 4), (1, 4), "[batch,out] / [1,out]"),
        ((2, 3, 4), (4,), "3D / 1D"),
        ((2, 3, 4), (3, 4), "3D / 2D"),

        # 4. Bidirectional broadcast
        ((3, 1), (1, 4), "[3,1] / [1,4] -> [3,4]"),
        ((1, 4), (3, 1), "[1,4] / [3,1] -> [3,4]"),

        # 5. Failure cases
        ((3,), (4,), "FAIL: incompatible dims"),
        ((2, 3), (3, 2), "FAIL: transposed"),
        ((2, 3), (4,), "FAIL: last dim mismatch"),
    ]

    print("\nSuccess Cases:")
    print("-" * 60)
    success_cases = []

    for shape_a, shape_b, desc in test_cases:
        result = test_broadcast_div(shape_a, shape_b)

        if result[0]:
            success, result_shape, a_data, b_data, result_data = result
            print(f"  OK: {list(shape_a)} / {list(shape_b)} -> {list(result_shape)} ({desc})")
            print(f"      a = {a_data}")
            print(f"      b = {b_data}")
            print(f"      r = {[round(x, 6) for x in result_data]}")
            success_cases.append((shape_a, shape_b, result_shape, a_data, b_data, result_data))
        else:
            print(f"  FAIL: {list(shape_a)} / {list(shape_b)} ({desc})")

    # Output Rust test code
    print("\n")
    print("=" * 70)
    print("Rust Test Code Template")
    print("=" * 70)

    print("""
/// / broadcast test cases (based on NumPy reference)
///
/// Note: Division is NOT commutative (a / b != b / a)
/// Note: Test data avoids zero divisors (Only Torch panics on div by zero)
///
/// Reference: tests/python/tensor_reference/tensor_div_broadcast_reference.py
#[test]
fn test_div_broadcast() {
    // Format: (shape_a, data_a, shape_b, data_b, expected_shape, expected_data)
    let test_cases: &[(&[usize], &[f32], &[usize], &[f32], &[usize], &[f32])] = &[""")

    for shape_a, shape_b, result_shape, a_data, b_data, result_data in success_cases:
        a_str = ", ".join(f"{x:.1f}" for x in a_data)
        b_str = ", ".join(f"{x:.1f}" for x in b_data)
        # Round to avoid floating point precision issues
        r_str = ", ".join(f"{round(x, 6)}" for x in result_data)

        print(f"        // {list(shape_a)} / {list(shape_b)} -> {list(result_shape)}")
        print(f"        (&{list(shape_a)}, &[{a_str}], &{list(shape_b)}, &[{b_str}], &{list(result_shape)}, &[{r_str}]),")

    print("""    ];

    for (shape_a, data_a, shape_b, data_b, expected_shape, expected_data) in test_cases {
        let a = Tensor::new(*data_a, *shape_a);
        let b = Tensor::new(*data_b, *shape_b);
        let expected = Tensor::new(*expected_data, *expected_shape);

        let result = &a / &b;
        assert_eq!(result, expected, "/ broadcast failed: {:?} / {:?}", shape_a, shape_b);

        // Note: Division is NOT commutative, so we don't test b / a here
    }
}""")


if __name__ == "__main__":
    main()
