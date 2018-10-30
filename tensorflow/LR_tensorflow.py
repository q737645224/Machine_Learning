import tensorflow as tf

#训练参数问题：trainable
#学习率和步数的设置：
#添加权重，损失值等在tensorboard观察的情况 1、收集变量 2、合并变量写入事件文件


def myregression():
    """
    自实现一个线性回归预测
    :return: None
    """
    with tf.variable_scope("data"):
        #1、准备数据，X特征值[100,1] 目标值[100]
        x = tf.random_normal([100, 1], mean=1.74, stddev=0.5, name="x_data")

        #矩阵相乘必须是二维的
        y_true = tf.matmul(x, [[0.7]]) + 0.8

    with tf.variable_scope("model"):
        #建立线性回归模型，1个特征，1个权重， 一个偏置 y=x w + b
        #随机给一个权重和偏置的值，让它去计算损失，然后在当前状态下优化
        weight = tf.Variable(tf.random_normal([1, 1], mean=0.0 , stddev=1.0) name='w')
        bias = tf.Variable(0.0, name='b')
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope("loss")
        #建立损失函数，均方误差
        loss = tf.reduce_mean(tf.square(y_true - y_predict))

    with tf.variable_scope("optimizer"):
        # 4.梯度下降优化损失 learning_rate:0~1, 2, 3, 4, 5,  6, 10
        tf.train.GradientDesecentOptimizer(0.1).minimize(loss)

    #1、收集tensor
    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights",weitht)

    #定义一个初始化变量op
    init_op = tf.global_variables_initializer()

    #通过回话运行程序
    with tf.Session() as sess:
        #运行初始化变量
        sess.run(init_op)

        #打印随机最先初始化的权重和偏置
        print('随机初始化参数为 ：%f, 偏置为 %f', %(weight.eval(), bias.eval()))

        #循环训练 运行优化
        for i in range(100):
            #运行优化
            sess.run(train_op)
            #运行合并的tensor
            summary = sess.run
            print("地%d参数权重为：%f, 偏置为：%f" %(i, weight.eval(), bias.eval()))
    return None

if __name__ == "__main__":
    myregression()