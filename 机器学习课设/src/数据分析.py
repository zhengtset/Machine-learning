import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import imblearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('D:\桌面\人工智能专业\实验\机器学习\机器学习课设\data\speed_dating_train.csv', encoding='gbk')

# 多少人通过Speed Dating找到了对象
plt.subplots(figsize=(5, 3), dpi=110, )
# 构造数据
size_of_groups = df.match.value_counts().values

single_percentage = round(size_of_groups[0] / sum(size_of_groups) * 100, 2)
matched_percentage = round(size_of_groups[1] / sum(size_of_groups) * 100, 2)
names = [
    'Single:' + str(single_percentage) + '%',
    'Matched' + str(matched_percentage) + '%']

# 创建饼图
plt.pie(
    size_of_groups,
    labels=names,
    labeldistance=1.2,
)
plt.show()

df[df.gender == 0]
# 多少女生通过Speed Dating找到了对象
plt.subplots(figsize=(5, 3), dpi=110, )
# 构造数据
size_of_groups = df[df.gender == 0].match.value_counts().values  # 男生只需要吧0替换成1即可

single_percentage = round(size_of_groups[0] / sum(size_of_groups) * 100, 2)
matched_percentage = round(size_of_groups[1] / sum(size_of_groups) * 100, 2)
names = [
    'Single:' + str(single_percentage) + '%',
    'Matched' + str(matched_percentage) + '%']
# 创建饼图
plt.pie(
    size_of_groups,
    labels=names,
    labeldistance=1.2,
)
plt.show()

# 年龄分布
age = df[np.isfinite(df['age'])]['age']
plt.hist(age,bins=35)
plt.xlabel('Age')
plt.ylabel('Frequency')

date_df = df[[
    'iid', 'gender', 'pid', 'match', 'int_corr', 'samerace', 'age_o',
       'race_o', 'pf_o_att', 'pf_o_sin', 'pf_o_int', 'pf_o_fun', 'pf_o_amb',
       'pf_o_sha', 'dec_o', 'attr_o', 'sinc_o', 'intel_o', 'fun_o', 'like_o',
       'prob_o', 'met_o', 'age', 'race', 'imprace', 'imprelig', 'goal', 'date',
       'go_out', 'career_c', 'sports', 'tvsports', 'exercise', 'dining',
       'museums', 'art', 'hiking', 'gaming', 'clubbing', 'reading', 'tv',
       'theater', 'movies', 'concerts', 'music', 'shopping', 'yoga', 'attr1_1',
       'sinc1_1', 'intel1_1', 'fun1_1', 'amb1_1', 'attr3_1', 'sinc3_1',
       'fun3_1', 'intel3_1', 'dec', 'attr', 'sinc', 'intel', 'fun', 'like',
       'prob', 'met'
]]
# heatmap
plt.subplots(figsize=(18,12))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = date_df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

clean_df = df[['attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o','match']]
clean_df.dropna(inplace=True)
X=clean_df[['attr_o','sinc_o','intel_o','fun_o','amb_o','shar_o',]]
y=clean_df['match']

oversample = imblearn.over_sampling.SVMSMOTE()
X, y = oversample.fit_resample(X, y)

# 做训练集和测试集分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
# 构建分类模型并进行训练
model = ExtraTreesClassifier()
model.fit(X_train, y_train)

# 预测训练集和测试集，计算准确率
predict_train_lrc = model.predict(X_train)
predict_test_lrc = model.predict(X_test)
print('Training Accuracy:', metrics.accuracy_score(y_train, predict_train_lrc))
print('Validation Accuracy:', metrics.accuracy_score(y_test, predict_test_lrc))

# 对测试集进行预测并输出结果到csv文件
predict_test = model.predict(X_test)
print(predict_test)

# 将该数组转换为DataFrame并从1开始编号
df = pd.DataFrame(predict_test, columns=["match"])
df.insert(0, 'uid', range(1, len(df) + 1))

# 将DataFrame写入CSV文件
df.to_csv("output.csv", index=False)