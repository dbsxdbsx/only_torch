# -*- coding: utf-8 -*-
"""
运行所有 Gymnasium 环境测试

使用方法:
    python tests/python/gym/run_all_tests.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import subprocess
import os

# 获取当前脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 测试脚本列表（按顺序执行）
TEST_SCRIPTS = [
    ("基础离散环境", "test_01_basic_discrete.py"),
    ("基础连续环境", "test_02_basic_continuous.py"),
    ("Box2D 环境", "test_03_box2d.py"),
    ("MuJoCo 环境", "test_04_mujoco.py"),
    ("Atari 环境", "test_05_atari.py"),
    ("Minari 数据集", "test_06_minari.py"),
    ("混合动作空间", "test_07_hybrid.py"),
]


def run_test(name: str, script: str) -> bool:
    """运行单个测试脚本"""
    script_path = os.path.join(SCRIPT_DIR, script)
    
    if not os.path.exists(script_path):
        print(f"⚠️  {name}: 脚本不存在 ({script})")
        return False
    
    print(f"\n{'='*60}")
    print(f"🔄 正在运行: {name}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )
    
    return result.returncode == 0


def main():
    print("\n" + "=" * 60)
    print("🚀 Gymnasium 环境完整测试套件")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, script in TEST_SCRIPTS:
        if run_test(name, script):
            passed += 1
        else:
            failed += 1
            print(f"❌ {name} 测试失败!")
    
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    print(f"  ✅ 通过: {passed}")
    print(f"  ❌ 失败: {failed}")
    print(f"  📝 总计: {len(TEST_SCRIPTS)}")
    
    if failed == 0:
        print("\n🎉 所有测试通过! 环境搭建成功!")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查相关依赖安装")
    
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
