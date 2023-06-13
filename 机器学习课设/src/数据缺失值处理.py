# 数据准备
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('D:\桌面\人工智能专业\实验\机器学习\机器学习课设\data\speed_dating_train.csv')
print(data.shape)

# 缺失值考察
def missing(data,threshold):
    percent_missing = data.isnull().sum() / len(data)
    missing = pd.DataFrame({'column_name': data.columns,'percent_missing': percent_missing})
    missing_show = missing.sort_values(by='percent_missing')
    print(missing_show[missing_show['percent_missing']>0].count())
    print('----------------------------------')
    out = missing_show[missing_show['percent_missing']>threshold]
    return out
missing(data,0.7)

missing(data,0.7).to_csv('第一轮缺省值处理.csv')

# 我们尝试分析X_2表单缺失，是否和匹配存在关系
def null_infomation(data,column):
    data_null = data[[column,'match']].dropna()
    data_shape = data.shape[0]
    data_null_shape = data_null.shape[0]
    print(f'{column}缺失{data_shape-data_null_shape}个值，缺失率为{100*(data_shape-data_null_shape)/data_shape}%')
    dif = 100*(data[data['match']==1].shape[0]/data_shape - data_null[data_null['match']==1].shape[0]/data_null_shape)
    print(f'样本的整体偏差率为{dif}%')
null_infomation(data,'attr7_2')

# 检查这个观点其实很简单，我们要检查是否存在全部缺失的样本是否可以成功匹配
data_2 = data.loc[:,'satis_2':'amb5_2']
data_2_null = data_2.dropna(how = 'all')
data_2.shape[0]-data_2_null.shape[0]
data[data.iid==8].loc[:,'satis_2':'amb5_2']
data[data.iid==8].loc[:,'round':'attr_o']
data[data.iid==12].loc[:,'round':'attr_o']
data[data.iid==12].loc[:,'satis_2':'amb5_2'].T
data_1 = data.loc[:,'iid':'amb3_s']
print(data_1.shape)

# 适当调低阈值进一步审查
missing(data_1,0.3)
missing(data_1,0.3).to_csv('第二轮缺省值处理.csv')


null_infomation(data_1,'expnum')
# 大学的SAT平均分，用来代表大学水平，缺失可能是较差的大学或者没有大学就读
# 这个偏差同样可观，而且偏差为正，缺失带来竞争劣势
null_infomation(data_1,'mn_sat')
# 本科生学费，影响不可观，删掉
null_infomation(data_1,'tuition')
# 后缀3_s，认为自己的吸引力，影响不可观，删掉
null_infomation(data_1,'amb3_s')
# 进行到一半，问吸引力侧重点，影响不可观，删掉
null_infomation(data_1,'shar1_s')
# 收入，暂时保存
null_infomation(data_1,'income')

# 删掉刚刚初步分析的几个组量，并且删去对明显没有用的、或者有编号的文字特征
data_1.drop(columns = ['tuition','tuition','attr3_s','sinc3_s','intel3_s','fun3_s','amb3_s',
                       'shar1_s','attr1_s','sinc1_s','intel1_s','fun1_s','amb1_s',
                       'position','positin1','field','from'],inplace=True)
print(data_1.shape)

# 适当调低阈值再进一步审查，实际上这是最后一次整体审查，之后小于10%的缺失值，将会采取策略填充
missing(data_1,0.1)
missing(data_1,0.1).to_csv('第三轮缺省值处理.csv')

# 这个偏差还是较小的（按照百分之五做阈值，即0.82%以上，稍稍超过）
null_infomation(data_1,'attr5_1')
# 本科毕业院校，这个缺失率比较本科SAT小一些，但是人家本科SAT分数影响更可观，而且，文本结构编号手段复杂，删掉，大学信息保留一个SAT招生分数够了
null_infomation(data_1,'undergra')
# 后缀4_1，影响不可观，删掉
null_infomation(data_1,'shar4_1')
# 后续的影响都不可观，全部删掉
null_infomation(data_1,'match_es')
null_infomation(data_1,'shar_o')
null_infomation(data_1,'zipcode')
null_infomation(data_1,'shar')

data_1.drop(columns = ['undergra','attr5_1','sinc5_1','intel5_1','fun5_1','amb5_1',
                       'shar4_1','sinc4_1','attr4_1','intel4_1','fun4_1','amb4_1',
                       'match_es','shar_o','zipcode','shar'],inplace=True)
print(data_1.shape)

missing(data_1,0.05)
missing(data_1,0.05).to_csv('空值填充处理.csv')

# 对于没有填写期望约会数目的，单独记作一类
# 对于没有SAT分数的，单独记作一类
# 重新考虑了一下，还是去掉收入,反正这个收入也是用邮编估计的
data_1['expnum'].fillna(-1, inplace=True)
data_1['mn_sat'].fillna(-1, inplace=True)
data_1.drop(columns = ['income'],inplace=True)
data_1[data_1.isnull().any(axis=1)].shape

# 准备对所有数据进行填充，填充主要有用众数、平均数、0等填充手段。我计划主要采用众数填充（这样不会影响按类填充结果）
# 当然，对72个损失中，不适合采用众数填充的，单独处理
missing_out = missing(data_1,0)['column_name']
print(missing_out.index)
# id,pid实际上不是输入特征
data_1.drop(columns = ['iid','pid'],inplace=True)
# 经过审查(随便扫一眼)，应该问题不大，都可以使用众数填充

for columname in data_1.columns:
    data_1[columname].fillna(data_1[columname].mode()[0],inplace=True)
# 填充完毕，缺失审查结束
missing(data_1,0)


