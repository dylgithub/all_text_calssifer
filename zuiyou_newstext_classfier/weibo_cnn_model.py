#encoding=utf-8
import tensorflow as tf
#用于产生权重向量
def W_generate(shape):
    inital=tf.truncated_normal(shape)
    return tf.Variable(inital)
#用于产生偏置值
def bias_generate(shape):
    inital=tf.constant(0.1,shape=[shape])
    return tf.Variable(inital)
'''
该类用于完成cnn模型的正向传播，返回一个正向传播的结果
sentence_length：每个句子的长度
vector_size：每个单词向量的维度
num_classes：分类的类别数
num_filters：卷积核的个数
filter_hs：所使用的各个高度的卷积核
'''
class cnn_model():
    def __init__(self,sentence_length,vector_size,num_filters,num_classes,filter_hs,input_x,keep_prob):
        self.sentence_length=sentence_length
        self.vector_size=vector_size
        self.num_filters=num_filters
        self.num_classes=num_classes
        self.filter_hs=filter_hs
        self.x=input_x
        self.keep_prob=keep_prob
    #建立模型，执行正向传播，返回正向传播得到的值
    def positive_propagation(self):
        #随后进行卷积操作
        print('begin conve')
        with tf.name_scope("conve"):
            W_conv = []
            b_conv = []
            for filter_h in self.filter_hs:
                W_conv1 = W_generate([filter_h, self.vector_size, 1, self.num_filters])
                b_conv1 = bias_generate(self.num_filters)
                W_conv.append(W_conv1)
                b_conv.append(b_conv1)
            con_outputs = []
            # print(np.shape(input_x))
            for W, b in zip(W_conv, b_conv):
                con_output = tf.nn.relu(tf.nn.conv2d(self.x,W,strides=[1,1,1,1],padding='VALID'))
                con_outputs.append(con_output)
        #再进行池化操作
        print('begin pool')
        with tf.name_scope("pool"):
            pool_outputs = []
            for con, filter_h in zip(con_outputs, self.filter_hs):
                pool_output = tf.nn.max_pool(con,ksize=[1,self.sentence_length-filter_h+1,1,1],strides=[1,1,1,1],padding='VALID')
                pool_outputs.append(pool_output)

        #全连接层操作
        # pool_outputs中的每一个元素的维度为(30, 1, 1, 128)
        print('begin full_connection')
        with tf.name_scope("full_connection"):
            h_pool = tf.concat(pool_outputs, 3)  # 把3种大小卷积核卷积池化之后的值进行连接
            num_filters_total = self.num_filters * len(self.filter_hs)
            # 因为随后要经过一个全连接层得到与类别种类相同的输出，而全连接接收的参数是二维的，所以进行维度转换
            h_pool_flaten = tf.reshape(h_pool, [-1, num_filters_total])
            h_drop = tf.nn.dropout(h_pool_flaten, self.keep_prob)
            W = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes]))
            l2_loss = tf.nn.l2_loss(W)
            # 注意必须是浮点型
            b = tf.Variable(tf.constant(0., shape=[self.num_classes]), name="b")
            y_pred = tf.nn.xw_plus_b(h_drop, W, b, name="scores")  # wx+b
        return y_pred,l2_loss

