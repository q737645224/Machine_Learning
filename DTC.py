import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
# import helpers_05_08

def Decision_tree():
    """
    scikit-learn中的make_blobs方法常被用来生成聚类算法的测试数据，
    直观地说，make_blobs会根据用户指定的特征数量、中心点数量、范围等来生成几类数据，这些数据可用于测试聚类算法的效果。
    测试决策树算法
    :return:
    """
    #定义一个辅助函数
    def visualize_classifier(model, X, y, ax=None, cmap="rainbow"):
        ax = ax or plt.gca()
        #画出训练数据
        ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')
        ax.axis('off')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        #用评估器拟合数据
        model.fit(X, y)
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                             np.linspace(*ylim, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        #为结果生成彩色图
        n_class = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3,
                               levels=np.arange(n_class + 1) - 0.5,
                               cmap=cmap, clim=(y.min(), y.max()),
                               zorder=1)
    #用make_blobs数据集
    X, y =make_blobs(n_samples=300, centers=5, random_state=1, cluster_std=1.0)
    visualize_classifier(DecisionTreeClassifier(), X, y)
    return None

def bagging(x, y):
    """
    #使用袋装分类器集成决策树算法
    :return: None
    """
    tree = DecisionTreeClassifier()
    bag = BaggingClassifier(tree, n_estimators=100, max_samples=0.8, random_state=0)
    visualize_classifier(bag, x, y)
    return None

def decision():
    """
    决策树对泰坦尼克号进行预测生死
    :return:None
    """
    #获取数据
    titan = pd.read_scv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    #处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']

    #使用平均值，填充缺失值
    x['age'].fillna(x['age'].mean(), inplace=True)

    #分割数据到训练集 测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    #进行特征工程 特征 》》类别  》 one_hot编码
    dict = DictVectorizer(sparse=Flase)
    #pd转换为字典
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.fit_transform(x_test.to_dict(orient="records"))
    print(x_train)

    #用决策树进行预测
    dec = DecisionTreeClassifier()
    dec.fit(x_train,y_train)
    #预测准确率
    print("预测的准确率：", dec.score(x_test, t_test))

    #导出决策树的结构
    exportgraphviz(dec, out_file="./tree.dot", feature_name=["age", 'pclass'])


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
    # model = RandomForestClassifier(n_estimators=100, criterion="gini",
    #                                max_depth=5, max_features='quto',
    #                                bootstrap=True,random_state=0)



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

    x_train, x_test, y_train, y_test = fearture_engineering()
    #随机森林进行预测（参数调优）
    rf = RandomForestClassifier()
    # 网格搜索与交叉验证 modol 参数字典
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    #网格搜索param_grid- dict, cv - 交叉轮次
    grid = GridSearchCV(rf, param_grid=param, cv=2)
    grid.fit(x_train, y_train)
    print("准确率", grid.score(x_test, y_test_))
    #获取最优参数
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


if __name__ ==" __main__":
    randomforest()