#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
R 语言环境检查脚本
用于验证 rpy2 和 R 包是否正确安装
"""

import sys


def check_python_packages():
    """检查 Python 包"""
    print("=" * 60)
    print("1. 检查 Python 包")
    print("=" * 60)
    
    packages = {
        'rpy2': '3.5.16',
        'pandas': None,
        'numpy': None,
        'sklearn': None
    }
    
    all_installed = True
    for package, expected_version in packages.items():
        try:
            if package == 'sklearn':
                module = __import__(package)
            else:
                module = __import__(package)
            
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package:12s} v{version}")
            
            if expected_version and not version.startswith(expected_version.split('.')[0]):
                print(f"   ⚠️  警告：预期版本 ~={expected_version}, 实际={version}")
                
        except ImportError:
            print(f"❌ {package:12s} 未安装")
            all_installed = False
    
    return all_installed


def check_r_environment():
    """检查 R 环境"""
    print("\n" + "=" * 60)
    print("2. 检查 R 环境")
    print("=" * 60)
    
    try:
        import rpy2.robjects as ro
        
        # 获取 R 版本
        r_version = ro.r('R.version.string')[0]
        print(f"✅ R 语言版本：{r_version}")
        
        # 获取 R 路径
        r_home = ro.r('R.home()')[0]
        print(f"✅ R 安装路径：{r_home}")
        
        return True
        
    except Exception as e:
        print(f"❌ R 环境检查失败：{e}")
        return False


def check_r_packages():
    """检查 R 包"""
    print("\n" + "=" * 60)
    print("3. 检查 R 包")
    print("=" * 60)
    
    r_packages = ['ggplot2', 'reshape2', 'grid', 'gridExtra', 'scales']
    
    try:
        import rpy2.robjects as ro
        
        all_installed = True
        for pkg in r_packages:
            try:
                # 尝试加载包
                ro.r(f'library({pkg}, character.only=TRUE)')
                print(f"✅ {pkg}")
            except Exception:
                print(f"❌ {pkg} 未安装")
                all_installed = False
        
        return all_installed
        
    except Exception as e:
        print(f"❌ R 包检查失败：{e}")
        return False


def test_rpy2_pandas_conversion():
    """测试 rpy2 和 pandas 的数据转换"""
    print("\n" + "=" * 60)
    print("4. 测试 rpy2-pandas 数据转换")
    print("=" * 60)
    
    try:
        import pandas as pd
        import numpy as np
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        
        # 激活转换（已弃用，使用 localconverter）
        
        # 创建测试数据
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
        
        # 转换到 R
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)
            ro.globalenv['test_df'] = r_df
        
        # 在 R 中读取
        result = ro.r('nrow(test_df)')[0]
        
        if result == 3:
            print("✅ pandas↔R 数据转换正常")
            return True
        else:
            print(f"❌ 数据转换异常：预期 3 行，实际{result}行")
            return False
            
    except Exception as e:
        print(f"❌ 数据转换测试失败：{e}")
        return False


def provide_installation_commands():
    """提供安装命令"""
    print("\n" + "=" * 60)
    print("📦 如需安装缺失的组件，请执行以下命令:")
    print("=" * 60)
    
    print("\n【1. 安装 Python 包】")
    print("pip install rpy2==3.5.16 pandas numpy scikit-learn")
    
    print("\n【2. 安装 R 包】")
    print("在 R 或 RStudio 中运行:")
    print("""
install.packages(c(
  "ggplot2",
  "reshape2", 
  "grid",
  "gridExtra",
  "scales"
))
""")
    
    print("\n【3. macOS 用户可能需要设置 R_HOME】")
    print("export R_HOME=$(R RHOME)")
    print("")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("🔍 R 语言可视化环境检查工具")
    print("=" * 60 + "\n")
    
    # 执行各项检查
    python_ok = check_python_packages()
    r_env_ok = check_r_environment()
    r_packages_ok = check_r_packages()
    conversion_ok = test_rpy2_pandas_conversion()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 检查结果汇总")
    print("=" * 60)
    
    all_ok = all([python_ok, r_env_ok, r_packages_ok, conversion_ok])
    
    if all_ok:
        print("✅ 所有检查通过！您可以使用 R 语言可视化功能了")
        print("\n运行以下命令开始生成热力图:")
        print("  python run_r_heatmap.py")
        print("或")
        print("  python test_r_heatmap.py")
    else:
        print("⚠️ 部分检查未通过，请先安装缺失的组件")
        provide_installation_commands()
    
    print("=" * 60 + "\n")
    
    return all_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
