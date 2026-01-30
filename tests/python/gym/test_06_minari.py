# -*- coding: utf-8 -*-
"""
Minari 离线强化学习数据集测试

Minari 是 D4RL 的官方继任者，由 Farama Foundation 维护。

需要安装: pip install minari
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import minari


def test_list_remote_datasets():
    """测试列出远程可用数据集"""
    print("=" * 60)
    print("测试列出远程数据集")
    print("=" * 60)
    
    try:
        remote_datasets = minari.list_remote_datasets()
        print(f"远程可用数据集数量: {len(remote_datasets)}")
        
        # 显示前 10 个数据集
        print("\n前 10 个数据集:")
        for i, (name, info) in enumerate(list(remote_datasets.items())[:10]):
            print(f"  {i+1}. {name}")
        
        print("✅ 列出远程数据集测试通过!\n")
        return True
    except Exception as e:
        print(f"⚠️  列出远程数据集失败 (可能需要网络): {e}\n")
        return False


def test_list_local_datasets():
    """测试列出本地已下载的数据集"""
    print("=" * 60)
    print("测试列出本地数据集")
    print("=" * 60)
    
    local_datasets = minari.list_local_datasets()
    print(f"本地已下载数据集数量: {len(local_datasets)}")
    
    if local_datasets:
        print("\n本地数据集:")
        for name, info in local_datasets.items():
            print(f"  - {name}")
    else:
        print("  (无本地数据集)")
    
    print("✅ 列出本地数据集测试通过!\n")
    return True


def test_download_and_load_dataset():
    """测试下载并加载一个小型数据集"""
    print("=" * 60)
    print("测试下载并加载数据集")
    print("=" * 60)
    
    # 使用一个较小的数据集进行测试
    dataset_id = "D4RL/pointmaze/umaze-v2"
    
    try:
        # 检查是否已下载
        local_datasets = minari.list_local_datasets()
        
        if dataset_id not in local_datasets:
            print(f"正在下载数据集: {dataset_id}")
            minari.download_dataset(dataset_id)
            print("下载完成!")
        else:
            print(f"数据集已存在: {dataset_id}")
        
        # 加载数据集
        print(f"\n加载数据集: {dataset_id}")
        dataset = minari.load_dataset(dataset_id)
        
        print(f"  总 episode 数: {dataset.total_episodes}")
        print(f"  总 step 数: {dataset.total_steps}")
        
        # 采样一个 episode
        episodes = dataset.sample_episodes(1)
        if episodes:
            ep = episodes[0]
            print(f"  采样 Episode 长度: {len(ep.observations)}")
            print(f"  观察形状: {ep.observations.shape if hasattr(ep.observations, 'shape') else 'N/A'}")
        
        print("✅ 下载并加载数据集测试通过!\n")
        return True
        
    except Exception as e:
        print(f"⚠️  下载/加载数据集失败 (可能需要网络): {e}\n")
        return False


def main():
    print("\n" + "=" * 60)
    print("Minari 离线 RL 数据集测试")
    print("=" * 60 + "\n")
    
    results = []
    
    # 测试列出本地数据集（不需要网络）
    results.append(("列出本地数据集", test_list_local_datasets()))
    
    # 测试列出远程数据集（需要网络）
    results.append(("列出远程数据集", test_list_remote_datasets()))
    
    # 测试下载并加载数据集（需要网络）
    results.append(("下载并加载数据集", test_download_and_load_dataset()))
    
    # 汇总结果
    print("=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    
    for name, ok in results:
        status = "✅" if ok else "⚠️ "
        print(f"  {status} {name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 所有 Minari 测试通过!")
    else:
        print("\n⚠️  部分测试未通过（可能需要网络连接）")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
