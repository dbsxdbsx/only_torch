"""
Tensor SubAssign (-=) Broadcast Reference Test

Test cases for in-place subtraction with broadcasting.

Rules for a -= b:
- Broadcasting is allowed IF the result shape equals a's original shape
- If b broadcasts to a shape different from a, the operation fails

Reference script for Rust unit tests.
"""

import numpy as np


def test_sub_assign(shape_a, shape_b):
    """
    Test if a -= b succeeds with broadcasting.

    Returns: (success, a_data, b_data, result_data) or (False, error_msg)
    """
    try:
        size_a = int(np.prod(shape_a)) if shape_a else 1
        size_b = int(np.prod(shape_b)) if shape_b else 1

        a = np.arange(10, 10 + size_a, dtype=np.float32).reshape(shape_a) if shape_a else np.array(10.0, dtype=np.float32)
        b = np.arange(1, size_b + 1, dtype=np.float32).reshape(shape_b) if shape_b else np.array(1.0, dtype=np.float32)

        a_before = a.copy()
        a -= b

        return (True, a_before.flatten().tolist(), b.flatten().tolist(), a.flatten().tolist())
    except ValueError as e:
        return (False, str(e), None, None)


def main():
    print("=" * 70)
    print("NumPy -= Broadcast Reference Test")
    print("=" * 70)

    # Test cases: (shape_a, shape_b, description)
    test_cases = [
        # 1. Same shape (always succeeds)
        ((), (), "same shape scalar"),
        ((3,), (3,), "same shape vector"),
        ((2, 3), (2, 3), "same shape matrix"),

        # 2. Scalar broadcast to higher dims (succeeds)
        ((3,), (), "vector -= scalar"),
        ((2, 3), (), "matrix -= scalar"),
        ((2, 3, 4), (), "3D -= scalar"),

        # 3. Lower dim broadcast to higher dim (succeeds)
        ((2, 3), (3,), "matrix -= vector (last dim)"),
        ((2, 3), (1, 3), "[2,3] -= [1,3]"),
        ((3, 4), (4,), "[batch,out] -= [out] (bias pattern)"),
        ((3, 4), (1, 4), "[3,4] -= [1,4]"),
        ((2, 3, 4), (4,), "3D -= 1D"),
        ((2, 3, 4), (3, 4), "3D -= 2D"),
        ((2, 3, 4), (1, 3, 4), "3D -= 3D with batch=1"),

        # 4. Broadcast with dim=1 expansion (succeeds)
        ((3, 4), (3, 1), "[3,4] -= [3,1] column broadcast"),
        ((2, 3, 4), (1, 1, 4), "3D -= 3D with multiple 1s"),

        # 5. Failure cases: would change a's shape
        ((), (3,), "FAIL: scalar -= vector"),
        ((3,), (2, 3), "FAIL: vector -= matrix"),
        ((2, 3), (4,), "FAIL: last dim mismatch"),
        ((2, 3), (3, 2), "FAIL: transposed shape"),
    ]

    print("\nSuccess Cases:")
    print("-" * 60)
    success_cases = []

    for shape_a, shape_b, desc in test_cases:
        result = test_sub_assign(shape_a, shape_b)

        if result[0]:
            success, a_data, b_data, result_data = result
            print(f"  OK: {list(shape_a)} -= {list(shape_b)} ({desc})")
            print(f"      a = {a_data}")
            print(f"      b = {b_data}")
            print(f"      r = {result_data}")
            success_cases.append((shape_a, shape_b, a_data, b_data, result_data))
        else:
            print(f"  FAIL: {list(shape_a)} -= {list(shape_b)} ({desc})")

    # Output Rust test code
    print("\n")
    print("=" * 70)
    print("Rust Test Code Template")
    print("=" * 70)

    print("""
/// -= broadcast test cases (based on NumPy reference)
///
/// Reference: tests/python/tensor_reference/tensor_sub_assign_broadcast_reference.py
#[test]
fn test_sub_assign_broadcast() {
    // Format: (shape_a, data_a, shape_b, data_b, expected_data)
    // Note: result shape is always same as shape_a
    let test_cases: &[(&[usize], &[f32], &[usize], &[f32], &[f32])] = &[""")

    for shape_a, shape_b, a_data, b_data, result_data in success_cases:
        a_str = ", ".join(f"{x:.1f}" for x in a_data)
        b_str = ", ".join(f"{x:.1f}" for x in b_data)
        r_str = ", ".join(f"{x:.1f}" for x in result_data)

        print(f"        // {list(shape_a)} -= {list(shape_b)}")
        print(f"        (&{list(shape_a)}, &[{a_str}], &{list(shape_b)}, &[{b_str}], &[{r_str}]),")

    print("""    ];

    for (shape_a, data_a, shape_b, data_b, expected_data) in test_cases {
        let mut a = Tensor::new(*data_a, *shape_a);
        let b = Tensor::new(*data_b, *shape_b);
        let expected = Tensor::new(*expected_data, *shape_a);  // shape unchanged

        a -= &b;
        assert_eq!(a, expected, "-= broadcast failed: {:?} -= {:?}", shape_a, shape_b);
    }
}""")


if __name__ == "__main__":
    main()
