import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump

# 1. 加载数据集
data = pd.read_csv('database.csv')  
X = data.filter(regex='x\d+')  
y = data.filter(regex='y\d+')  

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 输入输出归一化处理
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

# 保存归一化模型
dump(x_scaler, 'x_scaler.pkl')
dump(y_scaler, 'y_scaler.pkl')

# 4. 定义模型和超参数网格
model = MLPRegressor(early_stopping=True, validation_fraction=0.2,random_state=42)
param_grid = {
    'hidden_layer_sizes': [(50,),(50, 50),(100, 50),(50, 100,50),(300,)], 
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.001,0.01,0.05,0.1],
    'max_iter': [300,500,800,1000,1500] 
}

# 5. 网格搜索
grid_search = GridSearchCV( estimator=model,param_grid=param_grid,cv=5,n_jobs=-1,scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train_scaled)
# 6. 获取最优模型
best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

# 7. 预测和反归一化
y_train_pred_scaled = best_model.predict(X_train_scaled)
y_test_pred_scaled = best_model.predict(X_test_scaled)

y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

# 8. 评估模型
def evaluate_model(y_true, y_pred, set_name):
    results = {}
    for i in range(y_true.shape[1]):
        r2 = r2_score(y_true.iloc[:, i], y_pred[:, i])
        mse = mean_squared_error(y_true.iloc[:, i], y_pred[:, i])
        results[f'y{i}'] = {'R2': r2, 'MSE': mse}
    return pd.DataFrame(results).T

train_results = evaluate_model(y_train, y_train_pred, 'Train')
test_results = evaluate_model(y_test, y_test_pred, 'Test')

# 保存评估结果
train_results.to_csv('train_results.txt')
test_results.to_csv('test_results.txt')

# 9. 绘制散点图
output_names = y.columns
plt.figure(figsize=(15, 20))
for i in range(y.shape[1]):
    plt.subplot(3, 2, i + 1)
    # 训练集
    train_label = f'Train (R²: {train_results.iloc[i, 0]:.3f}, MSE: {train_results.iloc[i, 1]:.3f})'
    plt.scatter(y_train.iloc[:, i], y_train_pred[:, i],
                c='blue', alpha=0.5, label=train_label)
    # 测试集
    test_label = f'Test (R²: {test_results.iloc[i, 0]:.3f}, MSE: {test_results.iloc[i, 1]:.3f})'
    plt.scatter(y_test.iloc[:, i], y_test_pred[:, i],
                c='red', alpha=0.5, label=test_label)
   
    min_val = min(y.iloc[:, i].min(), y_train_pred[:, i].min())
    max_val = max(y.iloc[:, i].max(), y_train_pred[:, i].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.title(f'{output_names[i]} Prediction vs True')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.legend()

plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300)
plt.show()

# 10. 保存模型
dump(best_model, 'ann_model.pkl')