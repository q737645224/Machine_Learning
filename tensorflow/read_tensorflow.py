import tensorflow as tf

#批处理大小，跟队列，数据的数量没有影响，只决定 这批次取多少数据
def csvread(filelist):
    """
    读取CSV文件
    :param filelist: 文件路径+名字的列表
    :return: 读取的内容
    """
    # 1、构造文件队列
    file_queue = tf.train.string_input_producer(filelist) #返回具有输出字符串的队列

    # 2、构造阅读器读取队列（按一行）
    reader = tf.TextLineReader() #返回读取器实例
    key , value = reader.read(file_queue)
    # print(value)

    # 3、对每行内容进行解码
    record_defaults:指定每个样本的每一列的类型，指定默认值
    records = [[2], [4.0]]
    example, label = tf.decode_csv(value, record_defaults=records)

    # print(example, label)

    #4、想要读取多个数据
    example_batch, label_batch = tf.train.batch([example, label], batch_size=9, num_threads=1, capacity=9)

    print(example_batch, label_batch)
    return example_batch, label_batch

def picread(filelist):
    """
    读取图片并转换成张量
    :param filelist: 文件路径 + 名字列表
    :return:每张图片的张量
    """
    #1、构造文件队列
    file_queue = tf.train.string_input_producer(filelist_)

    # 2、构造阅读器去读取图片内容（默认的要求有一张图片）
    reader = tf.WholeFileReader()

    key,value = reader.read(file_queue)
    print(value)

    #3、对读取的图片数进行解码
    image = tf.image.decode_jpeg(value)

    return None

#定义cifar的数据等命令行参数
flags = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("cifar_dir","./data/cifar10/cifar-10-batches-bin", "文件的目录")



class CifarRead(object):
    """
    完成读取二进制文件写进tfrecords,读取tfrecords
    """
    def __init__(self,filelist):
        #文件列表
        self.file_list = filelist

        #定义读取的图片的一些属性
        self.height = 32
        self.width =32
        self.channerl = 3

        #二进制文件每张图片的字节
        self.label_bytes = 1
        self.image_bytes =self.height * self.width * self.channerl
        self.image_bytes = self.label_bytes + self.image_bytes


    def read_and_decode(self):
        #1、构造文件队列
        tf.train.string_input_producer(self.file_list)

        #2、构造二进制文件读取器，读取内容
        reader = tf.FixedLengthRecordReader()
        key, value = reader.read(file_queue)
        #3、解码内容,二进制文件内容
        # print(value)
        label_image = tf.decode_raw(value, tf.uint8)
        # print(label_image)

        #4、处理分割出图片和标签数据，切除特征值和目标值  tf.cast() 转换数据类型
        label = tf.cast(tf.slice(label_image, [0], [self.label_bytes]), tf.int32)
        image = tf.slice(label_image, [self.label_bytes],[self.image_bytes])
        # print(label, image)

        #5、可以对图片的特征数据进行形状的改变你[3072]-->[32, 32, 3]
        image_reshape = tf.reshape(image,[self.height, self,width, self,channel])
        # print(label, image_reshape)

        #6、批处理数据
        image_batch, label_batch = tf.train.batch([image_reshape, label], batch_size=10, num_threads=1, capacity=9)
        print(image_batch, label_batch)
        return image_batch, label_batch


    def write_to_tfrecords(self, image_batcj, label_batch):
        """
        将图片的特征值和目标值存进tfrecords
        :param image_batcj: 10张图片的特征值
        :param label_batch: 10张图片的目标值
        :return: None
        """
        #1、构造一个tfrecords文件，建立存储器
        tf.python_io.TFRecordWriter(FLAGS.cifar_tfrecords)
        #2、循环将所有样本写入文件，每张图片样本都要构造example协议
        for i in range(10):
            #取出第i个图片数据的特征值和目标值
            image = image_batch[i].eval().tostring()

            label = label_batch[i].eval()[0]

            #构造一个样本的example
            example = tf.train.Example(features=tf.train.Featues(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))

            #写入单独的样本
            writer.write(example.SerializeToStrint())
        #关闭
        writer.close()
    def read_from_tfreads(self):
        #1、构造文件队列
        file_queue = tf.train.string_input_producer([FLAGS.cifar_tfrecords])

        #2、构造文件阅读器，阅读的内容example,value=一个样本的序列化example
        reader = tf.TfRecordReader()
        key, value = reder.read(file_queue)

        #3、解析example
        features = tf.parse_single_example(value, features={
           "image":tf.FixedLenFeature([], tf.string)
            "label":tf.FixLenFeature([],tf.int64)
        })
        # print(features['image'],features['label'])

        #4、解码内容,如果读取内容格式是string需要解码，如果是Int64,float32不需要解码
        image = tf.decode_raw(features['image'],tf.uint8)
        #固定图片的形状，方便与批处理
        tf.reshape(image, [self.height, self.width, self.channel])

        label = tf.cast(features["label"], tf.int32)

        print(image, label)
        #进行批处理
        image_batch, label_batch = tf.train.batch([image_reshape, label],batch_size=10, num_threads=1, capacity=10)

        return image_batch, label_batch





if __name__ == "__main__":
    #1、找到文件，放入列表   路径+名字  ——》列表当中
    file_name = os.listdir("FLAGS.cifar_dir")
    filelist = [os.path.join("./data/dog/", file) for file in file_name if file[-3:] == "bin"]
    # print(file_name)

    # 2、调用读取函数
    cf = CifarRead(filelist)
    image_batch, label_batch = cf.read_and_decode()

    #读取tfrecord内容
    image_batch, label_batch= tf.read_from_tfreads()
    with tf.Session() as sess:
        #开启读取文件协调器 coordinator（协调器）
        coord = tf.train.Coordinator() #返回协调器实例

        #开启读文件的线程
        threads = tf.train.start_queue_runners(sess,coord=coord)

        # 存进tfrecords文件
        print("开始存储")
        cf.write_to_tfrecords(image_batch, label_batch)

        #打印读取的内容
        print(sess.run([image_batch, label_batch]))

        #回收子线程
        coord.request_stop()
        coord.join(threads)





# if __name__ == "__main__":
#     #找到文件，放入列表   路径+名字  ——》列表当中
#     file_name = os.listdir("./data/csvdata")
#     filelist = [os.path.join("./data/csvdata/", file) for file in file_name]
#     print(file_name)
#     #调用读取函数
#     example_batch, label_batch = csvread(filelist)
#
#     with tf.Session() as sess:
#         #开启读取文件协调器 coordinator（协调器）
#         coord = tf.train.Coordinator() #返回协调器实例
#
#         #开启读文件的线程
#         threads = tf.train.start_queue_runners(sess,coord=coord)
#
#


#         #打印读取的内容
#         print(sess.run([example_batch, label_batch]))
#
#         #回收子线程
#         coord.request_stop()
#
#         coord.join(threads)


