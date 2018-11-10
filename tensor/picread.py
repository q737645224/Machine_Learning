import tensorflow as tf
import os

# 模拟一下同步先处理数据，然后才能取数据训练
# tensorflow当中，运行操作有依赖性

# # 1、首先定义队列
# Q = tf.FIFOQueue(3, tf.float32)
#
# # 放入一些数据
# enq_many = Q.enqueue_many([[0.1, 0.2, 0.3], ])
#
# # 2、定义一些处理数据的螺距，取数据的过程      取数据，+1， 入队列
#
# out_q = Q.dequeue()
#
# data = out_q + 1
#
# en_q = Q.enqueue(data)
#
# with tf.Session() as sess:
#     # 初始化队列
#     sess.run(enq_many)
#
#     # 处理数据
#     for i in range(100):
#         sess.run(en_q)
#
#     # 训练数据
#     for i in range(Q.size().eval()):
#         print(sess.run(Q.dequeue()))


# 模拟异步子线程 存入样本， 主线程 读取样本

# # 1、定义一个队列，1000
# Q = tf.FIFOQueue(1000, tf.float32)
#
# # 2、定义要做的事情 循环 值，+1， 放入队列当中
# var = tf.Variable(0.0)
#
# # 实现一个自增  tf.assign_add
# data = tf.assign_add(var, tf.constant(1.0))
#
# en_q = Q.enqueue(data)
#
# # 3、定义队列管理器op, 指定多少个子线程，子线程该干什么事情
# qr = tf.train.QueueRunner(Q, enqueue_ops=[en_q] * 2)
#
# # 初始化变量的OP
# init_op = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     # 初始化变量
#     sess.run(init_op)
#
#     # 开启线程管理器
#     coord = tf.train.Coordinator()
#
#     # 真正开启子线程
#     threads = qr.create_threads(sess, coord=coord, start=True)
#
#     # 主线程，不断读取数据训练
#     for i in range(300):
#         print(sess.run(Q.dequeue()))
#
#     # 回收你
#     coord.request_stop()
#
#     coord.join(threads)


# 批处理大小，跟队列，数据的数量没有影响，只决定 这批次取多少数据

def picread(filelist):
    """
    读取狗图片并转换成张量
    :param filelist: 文件路径+ 名字的列表
    :return: 每张图片的张量
    """
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(filelist)

    # 2、构造阅读器去读取图片内容（默认读取一张图片）
    reader = tf.WholeFileReader()

    key, value = reader.read(file_queue)

    print(value)

    # 3、对读取的图片数据进行解码
    image = tf.image.decode_jpeg(value)

    print(image)

    # 5、处理图片的大小（统一大小）
    image_resize = tf.image.resize_images(image, [200, 200])

    print(image_resize)

    # 注意：一定要把样本的形状固定 [200, 200, 3],在批处理的时候要求所有数据形状必须定义
    image_resize.set_shape([200, 200, 3])

    print(image_resize)

    # 6、进行批处理
    image_batch = tf.train.batch([image_resize], batch_size=20, num_threads=1, capacity=20)

    print(image_batch)

    return image_batch