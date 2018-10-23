from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor ,Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib
def mylinear():
    """
    线性回归直接预测房子价格
    :return:None
    """
    lb = load_boston()
    #分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    #标准化处理（目标值要不要标准化处理？）
    #特征值和目标值都是必须标准化处理，实例化两个标准化API

    #实例化
    std_x = StandardScaler()
    #将特征值和目标值都进行标准化

    #特征值
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    #目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    def LRgression():
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        print(lr.coef_)
        # 使用正规方程，预测测试集的房子价格
        y_lr_predict = std_y.inverse_transform(lr.predict(x_test))  #将标准化特征值回转
        print("预测房子的价格", y_lr_predict)
        print("正规方程的均方误差", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
        return None

    def sgdRegressor():
        sgd = SGDRegressor()
        sgd.fit(x_train, y_train)
        print(sgd.coef_)
        # 使用线性梯度下降，预测测试集的房子价格
        y_sgd_predict = std_y.inverse_transform(sgd.predict(x_test))
        print("预测房子的价格", y_sgd_predict)
        print("梯度下降均方误差", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))
        return None

    def ridge():
        rd = Ridge(alpha=1.0)
        rd.fit(x_train, y_train)
        print(rd.coef_)

        #使用L2正则化（岭回归），预测测试集的房子价格
        y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
        print("预测房子的价格", y_rd_predict)
        print("均方误差",mean_squared_error(std_y.inverse_transform(y_test),y_rd_predict))
        return None

    # #保存训练好的模型
    # joblib.dump(rd, './temp/test.pkl')
    # #加载模型
    # model = joblib.load('./temp/test.pkl')
    # y_predict = std_y.inverse_transform(model.predict(x_test))
    # print("保存模型的预测的结果", y_predict)

    LRgression()

    sgdRegressor()

    ridge()
    return None

if __name__ == "__main__":
    mylinear()