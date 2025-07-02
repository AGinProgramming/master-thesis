import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 示例数据集
num_samples = 35
num_features = 181

# 随机生成示例数据（实际数据应为你自己的数据集）
np.random.seed(42)  # 为了结果可重复
data = np.random.rand(num_samples, num_features)
labels = np.random.randint(0, 2, num_samples)  # 假设是二分类问题

# 5-fold 交叉验证
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 存储每个折的性能
performance = []

for train_index, test_index in kf.split(data):
    train_data, test_data = data[train_index], data[test_index]
    train_labels, test_labels = labels[train_index], labels[test_index]
    
    # 训练模型（示例使用逻辑回归，实际中可替换为你需要的模型）
    model = LogisticRegression()
    model.fit(train_data, train_labels)
    
    # 预测
    predictions = model.predict(test_data)
    
    # 计算准确性（或其他性能指标）
    accuracy = accuracy_score(test_labels, predictions)
    performance.append(accuracy)

# 计算平均性能
mean_performance = np.mean(performance)
print(f'Mean accuracy over 5 folds: {mean_performance}')
