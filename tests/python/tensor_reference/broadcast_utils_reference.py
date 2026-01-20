#!/usr/bin/env python3
"""
Broadcast utility functions reference test.

Tests for:
1. broadcast_shape(shape_a, shape_b) -> output_shape or None
2. sum_to_shape(tensor, target_shape) -> tensor with target_shape

Run: python broadcast_utils_reference.py
"""

import numpy as np


def test_broadcast_shape():
    """Test broadcast_shape function - compute output shape from two input shapes."""
    print("=" * 60)
    print("broadcast_shape(shape_a, shape_b) -> output_shape")
    print("=" * 60)

    # Test cases: (shape_a, shape_b, expected_output_shape or None)
    test_cases = [
        # 1. Same shape
        ([3, 4], [3, 4], [3, 4]),
        ([2, 3, 4], [2, 3, 4], [2, 3, 4]),
        ([], [], []),  # scalar + scalar

        # 2. Scalar broadcast
        ([], [3], [3]),
        ([3], [], [3]),
        ([], [2, 3], [2, 3]),
        ([2, 3], [], [2, 3]),

        # 3. Lower dim broadcast to higher dim
        ([3, 4], [4], [3, 4]),
        ([4], [3, 4], [3, 4]),
        ([2, 3, 4], [4], [2, 3, 4]),
        ([2, 3, 4], [3, 4], [2, 3, 4]),

        # 4. Broadcast with 1s
        ([3, 4], [1, 4], [3, 4]),
        ([3, 4], [3, 1], [3, 4]),
        ([3, 1], [1, 4], [3, 4]),  # bidirectional
        ([1, 4], [3, 1], [3, 4]),  # bidirectional

        # 5. Higher dimensional
        ([2, 3, 4], [1, 3, 1], [2, 3, 4]),
        ([2, 1, 4], [1, 3, 1], [2, 3, 4]),  # bidirectional
        ([1, 1, 4], [2, 3, 1], [2, 3, 4]),

        # 6. Incompatible shapes (should return None)
        ([3], [4], None),
        ([2, 3], [3, 2], None),
        ([2, 3], [4], None),
        ([2, 3, 4], [2, 5, 4], None),
    ]

    print("\nSuccess cases:")
    print("-" * 60)
    for shape_a, shape_b, expected in test_cases:
        if expected is not None:
            # NumPy broadcast
            try:
                a = np.zeros(shape_a) if shape_a else np.array(0.0)
                b = np.zeros(shape_b) if shape_b else np.array(0.0)
                result = np.broadcast_shapes(a.shape, b.shape)
                result = list(result)
                assert result == expected, f"Mismatch: {result} != {expected}"
                print(f"  {shape_a} + {shape_b} -> {result}")
            except ValueError as e:
                print(f"  ERROR: {shape_a} + {shape_b} raised {e}")

    print("\nFailure cases (should return None):")
    print("-" * 60)
    for shape_a, shape_b, expected in test_cases:
        if expected is None:
            try:
                a = np.zeros(shape_a) if shape_a else np.array(0.0)
                b = np.zeros(shape_b) if shape_b else np.array(0.0)
                result = np.broadcast_shapes(a.shape, b.shape)
                print(f"  ERROR: {shape_a} + {shape_b} should fail but got {list(result)}")
            except ValueError:
                print(f"  {shape_a} + {shape_b} -> None (incompatible)")

    # Generate Rust test data
    print("\n" + "=" * 60)
    print("Rust test data for broadcast_shape:")
    print("=" * 60)
    print("let success_cases: &[(&[usize], &[usize], &[usize])] = &[")
    for shape_a, shape_b, expected in test_cases:
        if expected is not None:
            print(f"    (&{shape_a}, &{shape_b}, &{expected}),")
    print("];")
    print("\nlet failure_cases: &[(&[usize], &[usize])] = &[")
    for shape_a, shape_b, expected in test_cases:
        if expected is None:
            print(f"    (&{shape_a}, &{shape_b}),")
    print("];")


def test_sum_to_shape():
    """Test sum_to_shape function - sum tensor along broadcast dimensions."""
    print("\n" + "=" * 60)
    print("sum_to_shape(tensor, target_shape)")
    print("=" * 60)

    # Test cases: (input_shape, target_shape, input_data, expected_data)
    test_cases = []

    # 1. Same shape (no-op)
    print("\n1. Same shape (no-op):")
    a = np.array([[1., 2., 3.], [4., 5., 6.]])  # [2, 3]
    target = [2, 3]
    result = a.copy()  # No change
    print(f"   [{list(a.shape)}] -> [{target}]")
    print(f"   Input:  {a.flatten().tolist()}")
    print(f"   Output: {result.flatten().tolist()}")
    test_cases.append((list(a.shape), target, a.flatten().tolist(), result.flatten().tolist()))

    # 2. Sum along axis 0: [32, 128] -> [1, 128]
    print("\n2. Sum along axis 0 (batch dimension):")
    a = np.arange(1, 13, dtype=np.float32).reshape(3, 4)  # [3, 4]
    target = [1, 4]
    result = a.sum(axis=0, keepdims=True)
    print(f"   [{list(a.shape)}] -> [{target}]")
    print(f"   Input:  {a.flatten().tolist()}")
    print(f"   Output: {result.flatten().tolist()}")
    test_cases.append((list(a.shape), target, a.flatten().tolist(), result.flatten().tolist()))

    # 3. Sum along axis 1: [3, 4] -> [3, 1]
    print("\n3. Sum along axis 1:")
    a = np.arange(1, 13, dtype=np.float32).reshape(3, 4)  # [3, 4]
    target = [3, 1]
    result = a.sum(axis=1, keepdims=True)
    print(f"   [{list(a.shape)}] -> [{target}]")
    print(f"   Input:  {a.flatten().tolist()}")
    print(f"   Output: {result.flatten().tolist()}")
    test_cases.append((list(a.shape), target, a.flatten().tolist(), result.flatten().tolist()))

    # 4. Sum to scalar: [3, 4] -> [1, 1]
    print("\n4. Sum to scalar:")
    a = np.arange(1, 13, dtype=np.float32).reshape(3, 4)  # [3, 4]
    target = [1, 1]
    result = a.sum(keepdims=True).reshape(1, 1)
    print(f"   [{list(a.shape)}] -> [{target}]")
    print(f"   Input:  {a.flatten().tolist()}")
    print(f"   Output: {result.flatten().tolist()}")
    test_cases.append((list(a.shape), target, a.flatten().tolist(), result.flatten().tolist()))

    # 5. Reduce dimension: [3, 4] -> [4] (sum axis 0, then squeeze)
    print("\n5. Reduce dimension count:")
    a = np.arange(1, 13, dtype=np.float32).reshape(3, 4)  # [3, 4]
    target = [4]
    result = a.sum(axis=0)
    print(f"   [{list(a.shape)}] -> [{target}]")
    print(f"   Input:  {a.flatten().tolist()}")
    print(f"   Output: {result.flatten().tolist()}")
    test_cases.append((list(a.shape), target, a.flatten().tolist(), result.flatten().tolist()))

    # 6. Multiple axes: [2, 3, 4] -> [1, 3, 1]
    print("\n6. Sum along multiple axes:")
    a = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)  # [2, 3, 4]
    target = [1, 3, 1]
    result = a.sum(axis=(0, 2), keepdims=True)
    print(f"   [{list(a.shape)}] -> [{target}]")
    print(f"   Input:  {a.flatten().tolist()}")
    print(f"   Output: {result.flatten().tolist()}")
    test_cases.append((list(a.shape), target, a.flatten().tolist(), result.flatten().tolist()))

    # 7. Higher dim: [2, 3, 4] -> [4]
    print("\n7. Higher dim to vector:")
    a = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)  # [2, 3, 4]
    target = [4]
    result = a.sum(axis=(0, 1))
    print(f"   [{list(a.shape)}] -> [{target}]")
    print(f"   Input:  {a.flatten().tolist()}")
    print(f"   Output: {result.flatten().tolist()}")
    test_cases.append((list(a.shape), target, a.flatten().tolist(), result.flatten().tolist()))

    # 8. [2, 3, 4] -> [3, 4]
    print("\n8. Remove first dimension:")
    a = np.arange(1, 25, dtype=np.float32).reshape(2, 3, 4)  # [2, 3, 4]
    target = [3, 4]
    result = a.sum(axis=0)
    print(f"   [{list(a.shape)}] -> [{target}]")
    print(f"   Input:  {a.flatten().tolist()}")
    print(f"   Output: {result.flatten().tolist()}")
    test_cases.append((list(a.shape), target, a.flatten().tolist(), result.flatten().tolist()))

    # Generate Rust test data
    print("\n" + "=" * 60)
    print("Rust test data for sum_to_shape:")
    print("=" * 60)
    print("let test_cases: &[(&[usize], &[usize], &[f32], &[f32])] = &[")
    for input_shape, target_shape, input_data, expected_data in test_cases:
        print(f"    // [{input_shape}] -> [{target_shape}]")
        print(f"    (&{input_shape}, &{target_shape}, &{input_data}, &{expected_data}),")
    print("];")


if __name__ == "__main__":
    test_broadcast_shape()
    test_sum_to_shape()
    print("\nAll tests passed!")
