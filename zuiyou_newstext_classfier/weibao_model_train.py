#encoding=utf-8
import tensorflow as tf
import numpy as np
import pickle as pickle
from zuiyou_newstext_classfier import weibo_data_helper,weibo_cnn_model
'''
进行反向传播，优化模型
'''
# 模型的超参数
tf.flags.DEFINE_integer("vector_size", 80, "每个单词向量的维度")
tf.flags.DEFINE_integer("sentence_length", 200, "设定的句子长度")
tf.flags.DEFINE_integer("num_filters", 128, "卷积核的个数")
tf.flags.DEFINE_integer("num_classes", 10, "类别种类数")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2正则化系数的比率")
filter_hs=[3,4,5]
# 训练参数
tf.flags.DEFINE_float("keep_prob", 0.6, "丢失率")
tf.flags.DEFINE_integer("batch_size", 128, "每个批次的大小")
tf.flags.DEFINE_integer("num_epochs", 10, "训练的轮数")
tf.flags.DEFINE_integer("num_steps", 100, "学习率衰减的步数")
tf.flags.DEFINE_float("init_learning_rate", 0.05, "初始学习率")
FLAGS = tf.flags.FLAGS
#因为这里直接获得向量不再进行look_embedding所以是float类型
x = tf.placeholder('float', [None, FLAGS.sentence_length,FLAGS.vector_size,1], name='input')
y = tf.placeholder('float', [None, FLAGS.num_classes], name='output')
keep_prob = tf.placeholder(tf.float32, name="keep_prob")
def backward_propagation():
    # 初始化模型
    cnn = weibo_cnn_model.cnn_model(FLAGS.sentence_length, FLAGS.vector_size, FLAGS.num_filters, FLAGS.num_classes,
                                   filter_hs, x, keep_prob)
    print('begin positive_propagation')
    y_pred, l2_loss = cnn.positive_propagation()
    yuce=tf.nn.softmax(y_pred)
    leibie=tf.arg_max(yuce,1)
    # 获得类别标签的one_hot编码
    _,label = weibo_data_helper.get_train_data()
    label = np.array(label)
    # 计算损失值
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred)
    losses = tf.reduce_mean(loss) + FLAGS.l2_reg_lambda * l2_loss

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # train-modle========================================
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(FLAGS.init_learning_rate, global_step, FLAGS.num_steps,
                                               0.01)  # 学习率递减
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(losses, global_step=global_step)
    #测试集最高准确率能够达到0.90
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #划分训练集和测试集，注意此处是单词的索引并不是单词对应的向量
        print('begin 获得训练集测试集')
        with open('pkl_model/data.pkl', 'rb') as fr:
            pkl_vec = pickle.load(fr)
        pkl_vec = np.expand_dims(pkl_vec, axis=-1)
        #划分训练集和测试集
        X_train = pkl_vec[:-5000]
        X_test = pkl_vec[-5000:]
        y_train = label[:-5000]
        y_test = label[-5000:]
        # #批量获得数据
        num_inter = int(len(y_train) / FLAGS.batch_size)
        for ite in range(FLAGS.num_epochs):
            for i in range(num_inter):
                start = i * FLAGS.batch_size
                end = (i + 1) * FLAGS.batch_size
                feed_dict={x:X_train[start:end],y:y_train[start:end],keep_prob:FLAGS.keep_prob}
                if i % 10 == 0:
                    train_accuracy = accuracy.eval(
                        feed_dict=feed_dict)
                    print("Epoch %d:Step %d accuracy is %f" % (ite,i, train_accuracy))
                sess.run(train_step, feed_dict=feed_dict)
        y_pred, yuce,leibie=sess.run([y_pred, yuce,leibie], feed_dict={x: X_test, y: y_test, keep_prob: 1.0})
        print("y_pred:",y_pred)
        print("yuce:",yuce)
        print("leibie:", leibie)
        print("test accuracy %g" % accuracy.eval(feed_dict={x:X_test,y:y_test,keep_prob:1.0}))
if __name__ == '__main__':
    backward_propagation()