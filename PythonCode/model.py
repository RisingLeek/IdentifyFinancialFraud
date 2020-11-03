import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = 'utf-8')

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

train_data = pd.read_csv('../Resource/finalData.csv', low_memory = False)
test_data = pd.read_csv('../Resource/finalData.csv', low_memory = False)

# 除去缺失数据过多的条目与变量
train_data.drop(['公司代码'], axis = 1, inplace = True)
train_data.drop(['日期'], axis = 1, inplace = True)
train_data.drop(['是否调整'], axis = 1, inplace = True)
train_data.drop(['利息保障倍数'], axis = 1, inplace = True)
train_data.drop(['长期资本收益率'], axis = 1, inplace = True)
train_data.drop(['a股代码'], axis = 1, inplace = True)
train_data.drop(['违规类型'], axis = 1, inplace = True)
train_data.drop(['查处年份'], axis = 1, inplace = True)
train_data.drop(['工业企业类型'], axis = 1, inplace = True)
train_data.dropna(axis=0, how='any', inplace = True)

test_data.drop(['公司代码'], axis = 1, inplace = True)
test_data.drop(['日期'], axis = 1, inplace = True)
test_data.drop(['是否调整'], axis = 1, inplace = True)
test_data.drop(['利息保障倍数'], axis = 1, inplace = True)
test_data.drop(['长期资本收益率'], axis = 1, inplace = True)
test_data.drop(['a股代码'], axis = 1, inplace = True)
test_data.drop(['违规类型'], axis = 1, inplace = True)
test_data.drop(['查处年份'], axis = 1, inplace = True)
test_data.drop(['工业企业类型'], axis = 1, inplace = True)
test_data.dropna(axis=0, how='any', inplace = True)

full_data = train_data # 完整数据

temp_data = train_data[train_data['是否违规'] == 0]
temp_data = temp_data.sample(n = 1795, replace = True, random_state = 1)
temp_data1 = train_data[train_data['是否违规'] == 1]
train_data = temp_data.append(temp_data1)
test_data = train_data

# 完整数据
full_data_x = full_data[['流动比率', '速动比率', '资产负债率',\
                        '经营活动现金流量总额/负债合计', '应收账款与收入比',\
                        '存货与收入比', '存货周转率', '流动资产与收入比',\
                        '总资产周转率', '其他应收款占流动资产比例',\
                        '资产报酬率', '总资产净利润率', '营业毛利率']]
full_data_y = full_data['是否违规']

# 测试模型所用数据
model_data = train_data[['流动比率', '速动比率', '资产负债率',\
                        '经营活动现金流量总额/负债合计', '应收账款与收入比',\
                        '存货与收入比', '存货周转率', '流动资产与收入比',\
                        '总资产周转率', '其他应收款占流动资产比例',\
                        '资产报酬率', '总资产净利润率', '营业毛利率', '是否违规']]
# 得出最终结果使用的预测数据
pre_data = test_data[['流动比率', '速动比率', '资产负债率',\
                    '经营活动现金流量总额/负债合计', '应收账款与收入比',\
                    '存货与收入比', '存货周转率', '流动资产与收入比',\
                    '总资产周转率', '其他应收款占流动资产比例',\
                    '资产报酬率', '总资产净利润率', '营业毛利率']]

select_x = model_data.drop('是否违规', axis = 1)
select_y = model_data['是否违规']
illegal_line = model_data[['是否违规']]
a = illegal_line['是否违规']
illegal_line = list(a)
feature_line = model_data.drop('是否违规', axis = 1)
# 随机选取80%为训练集，20%为测试集
x_train, x_test, y_train, y_test = train_test_split(feature_line, illegal_line, test_size = 0.2, random_state = 1)

# 模型包调用
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold,cross_val_score

# 设置10折交叉
kfold=StratifiedKFold(n_splits=10)
# 模型集合
models=[SVC(),
    DecisionTreeClassifier(),
    ExtraTreesClassifier(),
    GradientBoostingClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    LogisticRegression(random_state=0, solver='lbfgs', max_iter=2000,
                       multi_class='multinomial'),
    LinearDiscriminantAnalysis()]
results=[]
for model in models:
    result=cross_val_score(model, feature_line, illegal_line, scoring = 'f1', cv=kfold, n_jobs=-1)#用F1得分作为参考、k折划分数据集
    results.append(result)

# 把模型得分和模型名称放同一个dataframe里面
score_mean=[]
for result in results:
    score_mean.append(result.mean())
score_board=pd.DataFrame({'model':['SVC',
                                'DecisionTreeClassifier',
                                'ExtraTreesClassifier',
                                'GradientBoostingClassifier',
                                'RandomForestClassifier',
                                'KNeighborsClassifier',
                                'LogisticRegression',
                                'LinearDiscriminantAnalysis'],'f1-score':score_mean})

print(score_board)
print('\n') # 输出各个模型及其得分

# # 决策树模型 学习曲线调整参数
# tr_entropy = []
# te_entropy = []

# tr_gini = []
# te_gini = []

# te_recall_entropy = []
# te_recall_gini = []

# for i in range(10):
#     CLF = DecisionTreeClassifier(random_state = 25, max_depth = i + 1, criterion = 'entropy')
#     res = CLF.fit(x_train, y_train)
#     res = pd.DataFrame(res.predict(x_test))
#     score_tr = CLF.score(x_train, y_train)
#     score_te = cross_val_score(CLF, x_test, y_test, cv = 10).mean()
#     tr_entropy.append(score_tr)
#     te_entropy.append(score_te)
#     te_recall_entropy.append(recall_score(y_test, res))

#     CLF = DecisionTreeClassifier(random_state = 25, max_depth = i + 1, criterion = 'gini')
#     res = CLF.fit(x_train, y_train)
#     res = pd.DataFrame(res.predict(x_test))
#     score_tr = CLF.score(x_train, y_train)
#     score_te = cross_val_score(CLF, x_test, y_test, cv = 10).mean()
#     tr_gini.append(score_tr)
#     te_gini.append(score_te)
#     te_recall_gini.append(recall_score(y_test, res))

# fig, (ax0, ax1) = plt.subplots(1,2, figsize=(18, 6))    

# ax0.plot(range(1,11),tr_entropy,color='r',label='train')
# ax0.plot(range(1,11),te_entropy,color='blue',label='test')
# ax0.set_xticks(range(1,11))
# ax0.set_title('entropy')
# ax0.legend()

# ax1.plot(range(1,11),tr_gini,color='r',label='train')
# ax1.plot(range(1,11),te_gini,color='blue',label='test')
# ax1.set_xticks(range(1,11))
# ax0.set_title('gini')
# ax1.legend()

# print('学习曲线调整后的决策树模型:\nentropy上的最好准确度为{}\n召回率为{}\nf1_score为{}\nmax_depth = {}\ngini上的最好准确度为{}\n召回率为{}\nf1_score为{}\nmax_depth = {}\n'.\
#       format(max(te_entropy), te_recall_entropy[te_entropy.index(max(te_entropy))], 0.7*max(te_entropy)+0.3*te_recall_entropy[te_entropy.index(max(te_entropy))],\
#       te_entropy.index(max(te_entropy)) + 1, max(te_gini),\
#       te_recall_gini[te_gini.index(max(te_gini))], 0.7*max(te_gini)+0.3*te_recall_gini[te_gini.index(max(te_gini))], te_gini.index(max(te_gini)) + 1))

# # 决策树模型 网格搜索调整参数

# gini_threholds = np.linspace(0, 0, 5, 20)
# parameters = {'criterion':('gini','entropy')
#               ,'splitter':('best','random')
#               ,'max_depth':[*range(2,5)]
#               ,'min_samples_leaf':[*range(1,10,2)]
#             # ,'min_impurity_decrease':np.linspace(0,0.5,20)
#              }
# CLF = DecisionTreeClassifier(random_state = 25)
# GS = GridSearchCV(CLF, parameters, cv = 10, n_jobs=-1)
# GS.fit(x_train, y_train)

# print(GS.best_params_)

# CLF = DecisionTreeClassifier(random_state = 25, criterion = 'gini',\
#                              max_depth = 4, min_samples_leaf = 5,\
#                              splitter = 'best')
# CLF = CLF.fit(x_train, y_train)
# res = pd.DataFrame(CLF.predict(x_test))

# print('网格搜索调整后的决策树模型:\n最好的准确度为{}\n召回率为{}\n精确度为{}\nf1_score为{}\n'.format(cross_val_score(CLF, x_test, y_test, cv = 10).mean(),\
#       recall_score(y_test, res), precision_score(y_test, res), 0.7*recall_score(y_test, res)+0.3*precision_score(y_test, res)))

# 随机森林模型 网格搜索调整参数
RFC = RandomForestClassifier(class_weight='balanced')
RFC_param = {"n_estimators": range(1,101,10), "max_features": range(1,8,1)}
# 根据f1分值选出最好参数
GS = GridSearchCV(RFC, RFC_param, cv = 10, scoring='f1', n_jobs = -1, verbose = 1)
GS.fit(feature_line, illegal_line)

print(GS.best_params_)

# 输入优化过的参数进行拟合
RFC = RandomForestClassifier(max_features = GS.best_params_['max_features'], n_estimators = GS.best_params_['n_estimators'], class_weight='balanced')
RFC_acc = cross_val_score(RFC, feature_line, illegal_line, cv = 10, n_jobs=-1).mean()
RFC = RFC.fit(x_train, y_train)
res_RFC = pd.DataFrame(RFC.predict(x_test))

print('网格搜索调整后的随机森林模型:\n最好的准确度为{}\n召回率为{}\n精确度为{}\nf1_score为{}\n'.format(RFC_acc,\
      recall_score(y_test, res_RFC), precision_score(y_test, res_RFC), 0.7*recall_score(y_test, res_RFC)+0.3*precision_score(y_test, res_RFC)))

# 梯度提升树模型 网格搜索调整参数
GBC = GradientBoostingClassifier()
GBC_param = {'loss': ['deviance'], 'n_estimators': [100,180,260], 'learning_rate': [0.1,0.05,0.01], 'max_depth': [5,8,10], 'min_samples_leaf': [75,100,125,150]}
# 根据f1分值选出最好参数
GS = GridSearchCV(GBC, GBC_param, cv=kfold, scoring='f1', n_jobs=-1, verbose=1)
GS.fit(feature_line, illegal_line)

print(GS.best_params_)

# 输入优化过的参数进行拟合
GBC = GradientBoostingClassifier(learning_rate = GS.best_params_['learning_rate'], loss = GS.best_params_['loss'], max_depth = GS.best_params_['max_depth'],\
 min_samples_leaf = GS.best_params_['min_samples_leaf'], n_estimators = GS.best_params_['n_estimators'])
GBC_acc = cross_val_score(GBC, feature_line, illegal_line, cv = 10, n_jobs=-1).mean()
GBC = GBC.fit(x_train, y_train)
res_GBC = pd.DataFrame(GBC.predict(x_test))

print('网格搜索调整后的梯度提升树模型:\n最好的准确度{}\n召回率{}\n精确度为{}\nf1_score为{}\n'.format(GBC_acc,\
      recall_score(y_test, res_GBC), precision_score(y_test, res_GBC), 0.7*recall_score(y_test, res_GBC)+0.3*precision_score(y_test, res_GBC)))

# 向量机模型 网格搜索调整参数
SVR = SVC(class_weight='balanced')
SVR_param = {'kernel':('rbf','linear'),'C':[0.1,0.5,1.0]}
GS = GridSearchCV(SVR, SVR_param, scoring='f1', n_jobs=-1)
GS = GS.fit(feature_line, illegal_line)

print(GS.best_params_)

SVR = SVC(kernel = GS.best_params_['kernel'], C = GS.best_params_['C'], class_weight='balanced')
SVR_acc = cross_val_score(SVC, feature_line, illegal_line, cv = 10, n_jobs=-1).mean()
SVR = SVR.fit(x_train, y_train)
res_SVR = pd.DataFrame(SVR.predict(x_test))

print('网格搜索调整后的向量机模型:\n最好的准确度{}\n召回率{}\n精确度为{}\nf1_score为{}\n'.format(SVR_acc,\
      recall_score(y_test, res_SVR), precision_score(y_test, res_SVR), 0.7*recall_score(y_test, res_SVR)+0.3*precision_score(y_test, res_SVR)))

# Logistics回归 网格搜索调整参数
LOR = LogisticRegression(class_weight='balanced')
LOR_param = {'penalty': ('l1', 'l2'),'C': (0.01, 0.1, 1, 10, 100, 1000)}
GS = GridSearchCV(LOR, LOR_param, verbose=0, scoring='f1', cv=5, n_jobs=-1)
GS = GS.fit(feature_line, illegal_line)

print(GS.best_params_)

LOR = LogisticRegression(penalty=GS.best_params_['penalty'], C=GS.best_params_['c'], class_weight='balanced')
LOR_acc = cross_val_score(LOR, feature_line, illegal_line, cv = 10, n_jobs=-1).mean()
LOR = LOR.fit(x_train, y_train)
res_LOR = pd.DataFrame(LOR.predict(x_test))

print('网格搜索调整后的Logistic模型:\n最好的准确度{}\n召回率{}\n精确度为{}\nf1_score为{}\n'.format(LOR_acc,\
      recall_score(y_test, res_LOR), precision_score(y_test, res_LOR), 0.7*recall_score(y_test, res_LOR)+0.3*precision_score(y_test, res_LOR)))

from sklearn.metrics import confusion_matrix #绘制混淆矩阵
print('梯度提升树的混淆矩阵为：\n',confusion_matrix(y_test,res_GBC, labels = [1, 0]))
print('随机森林的混淆矩阵为：\n',confusion_matrix(y_test,res_RFC, labels = [1, 0]))
print('向量机的混淆矩阵为：\n',confusion_matrix(y_test,res_SVR, labels = [1, 0]))
print('Logististcs回归的混淆矩阵为：\n',confusion_matrix(y_test,res_LOR, labels = [1, 0]))

import joblib
# 导出训练好的模型
joblib.dump(RFC,  "../PythonModel/RFC.pkl")
joblib.dump(GBC,  "../PythonModel/GBC.pkl")
joblib.dump(SVR,  "../PythonModel/SVR.pkl")
joblib.dump(LOR,  "../PythonModel/LOR.pkl")

# 模型融合
from mlxtend.classifier import StackingClassifier # 导入stacking包

# 使用前面分类器产生的特征输出作为最后总的meta-classifier的输入数据
SCLF1 =StackingClassifier(classifiers=[RFC, GBC, SVR, LOR], 
                            meta_classifier=LogisticRegression())
                
# 使用第一层基本分类器产生的类别概率值作为meta-classfier的输入
SCLF2 =StackingClassifier(classifiers=[RFC, GBC, SVR, LOR],
                            use_probas=True,
                            average_probas=False, 
                            meta_classifier=LogisticRegression())

# 评估
result1=cross_val_score(SCLF1, x_train, y_train, scoring = 'f1', cv=kfold, n_jobs=-1)
result2=cross_val_score(SCLF2, x_train, y_train, scoring = 'f1', cv=kfold, n_jobs=-1)

if (result1.mean() >= result2.mean()):
    SCLF1_acc = cross_val_score(SCLF1, feature_line, illegal_line, cv = 10, n_jobs=-1).mean()
    SCLF1.fit(x_train, y_train)
    res_SCLF = pd.DataFrame(LOR.predict(x_test))
    print('四个融合后的模型:\n最好的准确度{}\n召回率{}\n精确度为{}\nf1_score为{}\n'.format(SCLF1_acc,\
          recall_score(y_test, res_SCLF), precision_score(y_test, res_SCLF), 0.7*recall_score(y_test, res_SCLF)+0.3*precision_score(y_test, res_SCLF)))
    print('四个融合后的混淆矩阵为：\n',confusion_matrix(y_test,res_SCLF, labels = [1, 0]))
    # 导出训练好的模型
    joblib.dump(SCLF1,  "../PythonModel/SCLF.pkl")
else:
    SCLF2_acc = cross_val_score(SCLF2, feature_line, illegal_line, cv = 10, n_jobs=-1).mean()
    SCLF2.fit(x_train, y_train)
    res_SCLF = pd.DataFrame(LOR.predict(x_test))
    print('四个融合后的模型:\n最好的准确度{}\n召回率{}\n精确度为{}\nf1_score为{}\n'.format(SCLF2_acc,\
          recall_score(y_test, res_SCLF), precision_score(y_test, res_SCLF), 0.7*recall_score(y_test, res_SCLF)+0.3*precision_score(y_test, res_SCLF)))
    print('四个融合后的混淆矩阵为：\n',confusion_matrix(y_test,res_SCLF, labels = [1, 0]))
    # 导出训练好的模型
    joblib.dump(SCLF2,  "../PythonModel/SCLF.pkl")