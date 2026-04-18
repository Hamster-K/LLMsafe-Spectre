import time
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest, f_regression, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

# start_time
start_time = time.time()
df = pd.read_excel('.xlsx')

X = df.iloc[:, :-1]  #
y = df.iloc[:, -1]   #

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

selector_var = VarianceThreshold(threshold=0.0)
X_var = selector_var.fit_transform(X)

remaining_feature_names = X.columns[selector_var.get_support()]

X_scaled = scaler.fit_transform(X_var)

selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X_scaled, y)

selected_features = remaining_feature_names[selector.get_support()].tolist()

# 获取F值
f_scores = selector.scores_

# 对应的特征名
feature_scores = list(zip(remaining_feature_names, f_scores))
feature_scores.sort(key=lambda x: x[1], reverse=True)


print("Top-5:")
for i, (name, score) in enumerate(feature_scores[:5], 1):
    print(f"{name}")
for i, (name, score) in enumerate(feature_scores[:5], 1):
    print(f"F_value: {score:.4f}")


for i, (name, score) in enumerate(feature_scores, 1):
    print(f"{i:2d}. {name:<30} F-score: {score:.4f}")

# 记录结束时间
end_time = time.time()

# 输出运行时间
print(f"runtime: {end_time - start_time:.2f} s")