# 这是库安装验证脚本，运行后会显示所有需要的库是否安装成功
import sys

print("="*50)
print("开始检查实验所需的库和环境...")
print("="*50)

# 1. 检查Python版本（要求3.x，最好3.8-3.10，太新可能不兼容）
print("\n1. 检查Python版本：")
python_version = sys.version
print(f"当前Python版本：{python_version.split()[0]}")
# 判断是否是3.x版本
if python_version.startswith("3."):
    print("✅ Python版本符合要求（3.x）")
else:
    print("❌ 错误！Python版本不是3.x，请重新安装Python 3.x（推荐3.8-3.10）")

# 2. 检查PyTorch（要求2.0.1）
print("\n2. 检查PyTorch：")
try:
    import torch
    torch_version = torch.__version__
    print(f"当前PyTorch版本：{torch_version}")
    # 验证是否能正常使用（比如检查CUDA，没有GPU也没关系）
    print(f"CUDA是否可用：{torch.cuda.is_available()}")  # 有显卡会显示True，没有显示False也不影响
    if torch_version.startswith("2.0.1"):
        print("✅ PyTorch版本符合要求（2.0.1）")
    else:
        print("⚠️  警告！PyTorch版本不是2.0.1，可能会有兼容性问题")
except ImportError:
    print("❌ 错误！没有安装PyTorch，请按照附录教程安装2.0.1版本")

# 3. 检查torchvision（要求0.15.2）
print("\n3. 检查torchvision：")
try:
    import torchvision
    tv_version = torchvision.__version__
    print(f"当前torchvision版本：{tv_version}")
    if tv_version.startswith("0.15.2"):
        print("✅ torchvision版本符合要求（0.15.2）")
    else:
        print("⚠️  警告！torchvision版本不是0.15.2，可能会有兼容性问题")
except ImportError:
    print("❌ 错误！没有安装torchvision，请按照附录教程安装0.15.2版本")

# 4. 检查numpy（实验必需）
print("\n4. 检查numpy：")
try:
    import numpy
    np_version = numpy.__version__
    print(f"当前numpy版本：{np_version}")
    print("✅ numpy已安装")
except ImportError:
    print("❌ 错误！没有安装numpy，请运行命令：pip install numpy")

# 5. 检查natsort（实验必需）
print("\n5. 检查natsort：")
try:
    import natsort
    ns_version = natsort.__version__
    print(f"当前natsort版本：{ns_version}")
    print("✅ natsort已安装")
except ImportError:
    print("❌ 错误！没有安装natsort，请运行命令：pip install natsort")

print("\n" + "="*50)
print("环境检查结束！")
print("✅ 所有库都显示“已安装”或“符合要求”，就可以继续下一步实验～")
print("❌ 如果有库显示“没有安装”，请按照提示的命令安装，或参考附录教程")
print("="*50)