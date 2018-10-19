import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction import DictVectorizer


def fearture_engineering():
    """
    数据进行缺失值填充
    字典特抽取
    :return: (x_train,x_test, y_train, y_test)
    """
    # 获取数据
    titan = pd.read_scv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']

    # 使用平均值，填充缺失值
    x['age'].fillna(x['age'].mean(), inplace=True)

    # 分割数据到训练集 测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行特征工程 特征 》》类别  》 one_hot编码
    dict = DictVectorizer(sparse=Flase)
    # pd转换为字典
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    x_test = dict.fit_transform(x_test.to_dict(orient="records"))
    #查看数据
    print(dict.get_feature_names())
    print(x_train)
    return (x_train,x_test, y_train, y_test)


def randomforest():
    """
    #使用集成算法里的随机森林评估器
    n_estimators: integer, optional(default=10)森林里的树木数量
	120， 200， 300， 500， 800， 1200
	criterion:string, 可选（default='gini')分割特征的测量方法
	max_depth: integer或None, 可选（默认=无）树的最大深度
	5, 8， 15， 25， 30
	max_features='auto'每个决策树的最大特征数量
	bootstrap: boolean, optional(default =True) 是否在构建树时使用放回抽样
    :return: None
    """
    x_train, x_test, y_train, y_test = fearture_engineering()
    # 随机森林进行预测（参数调优）
    rf = RandomForestClassifier()
    # 网格搜索与交叉验证 modol 参数字典
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网格搜索param_grid- dict, cv - 交叉轮次
    grid = GridSearchCV(rf, param_grid=param, cv=2)
    grid.fit(x_train, y_train)
    print("准确率", grid.score(x_test, y_test_))
    # 获取最优参数
    print("查看选择的参数模型：", grid.best_params_)

    # #用最优参数的模型模拟数据，并画图显示
    # model = grid.best_estimator_
    #
    # plt.scatter(x.ravel(), y)
    # lim = plt.axis()
    # y_test = model.fit(x, y).predict(x_test)
    # plt.plot(x_test.tavel(), y_test, hold=True)
    # plt.axis(lim)
    # plt.show()
    return None

randomforest()