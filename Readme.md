# 神经网络回归模型项目

## 项目概述
本项目实现了一个基于多层感知机(MLP)的回归模型，用于多目标预测任务。项目包含完整的数据处理流程、模型训练与调优、评估指标计算以及结果可视化功能。通过网格搜索技术自动优化神经网络超参数，并保存最佳模型用于后续预测。

## 主要功能
- 数据预处理：自动识别特征(x)和目标变量(y)列
- 数据标准化：对特征和目标变量分别进行标准化处理
- 超参数调优：使用网格搜索寻找最优神经网络结构
- 模型评估：计算R²和MSE评估指标
- 结果可视化：生成预测值与真实值的对比散点图
- 模型持久化：保存训练好的模型和标准化器

## 文件结构
```
project/
├── daima.py                  # 主程序脚本
├── database.csv  # 输入数据集
├── ann_model.pkl             # 训练好的神经网络模型
├── x_scaler.pkl              # 特征标准化器
├── y_scaler.pkl              # 目标变量标准化器
├── train_results.txt         # 训练集评估结果
├── test_results.txt          # 测试集评估结果
└── scatter_plots.png         # 预测结果可视化图
```

## 环境依赖
运行此项目需要以下Python库：
- Python 3.9+
- pandas
- numpy
- matplotlib
- scikit-learn
- joblib

安装所有依赖：
```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## 使用说明
1. 准备数据：
   - 确保数据集文件名为`database.csv`
   - 特征列命名格式：`x0`, `x1`, ..., `xn`
   - 目标列命名格式：`y0`, `y1`, ..., `ym`

2. 运行程序：
```bash
python daima.py
```

3. 输出结果：
   - `ann_model.pkl`: 训练好的神经网络模型
   - `x_scaler.pkl`, `y_scaler.pkl`: 标准化转换器
   - `train_results.txt`, `test_results.txt`: 评估指标
   - `scatter_plots.png`: 预测结果可视化

## 模型配置
网格搜索的超参数配置：
```python
param_grid = {
    'hidden_layer_sizes': [(50,), (50,50), (100,50), (50,100,50), (300,)], 
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.001, 0.01, 0.05, 0.1],
    'max_iter': [300, 500, 800, 1000, 1500]
}
```

## 结果解读
### 评估指标
- **R² (决定系数)**：越接近1表示模型拟合越好
- **MSE (均方误差)**：值越小表示预测越准确

### 可视化说明
`scatter_plots.png`包含以下信息：
- 蓝色点：训练集预测结果
- 红色点：测试集预测结果
- 黑色虚线：理想预测线(y=x)
- 每个子图标题对应不同目标变量(y0, y1, ...)
- 图例包含该变量的R²和MSE值

## 自定义配置
可调整的关键参数：
```python
# 数据分割比例
test_size=0.2  # 测试集比例
# 网格搜索设置
cv=5           # 五折交叉验证
n_jobs=-1      # 使用所有CPU核心
scoring='neg_mean_squared_error'  # 评估指标
```

## 注意事项
1. 确保输入数据格式正确：
   - 特征列名格式：`x0`, `x1`, ..., `xn`
   - 目标列名格式：`y0`, `y1`, ..., `ym`
   
2. 首次运行时程序会：
   - 自动进行数据标准化
   - 执行网格搜索（耗时较长）
   - 生成所有输出文件

3. 后续使用可直接加载保存的模型：
```python
from joblib import load

model = load('ann_model.pkl')
x_scaler = load('x_scaler.pkl')
y_scaler = load('y_scaler.pkl')

# 对新数据进行预测
new_data_scaled = x_scaler.transform(new_data)
predictions_scaled = model.predict(new_data_scaled)
predictions = y_scaler.inverse_transform(predictions_scaled)
```

## 性能优化
- 网格搜索使用并行计算(`n_jobs=-1`)
- 启用早期停止(`early_stopping=True`)防止过拟合
- 验证集比例设置(`validation_fraction=0.2`)
- 使用Adam优化器加速收敛

此项目提供了一个完整的神经网络回归建模流程，适用于各种回归预测任务，特别是多目标预测场景。