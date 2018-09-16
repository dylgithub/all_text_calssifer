#encoding=utf-8
import tensorflow as tf
import numpy as np
import time
from zuiyou_newstext_classfier import weibo_data_helper
import pickle as pickle
from six.moves import xrange
#0.93850
# 模型的超参数
tf.flags.DEFINE_integer("vector_size", 80, "每个单词向量的维度")
tf.flags.DEFINE_integer("sentence_length", 200, "设定的句子最大长度")
tf.flags.DEFINE_integer("n_hidden", 100, "隐藏层细胞的个数")
tf.flags.DEFINE_integer("attn_size", 200, "注意力层，此参数和lstm隐藏层大小相同，双向则为其2倍")
tf.flags.DEFINE_integer("num_classes", 10, "类别种类数")
# 训练参数
tf.flags.DEFINE_integer("batch_size", 64, "每个批次的大小")
#output_keep_prob=0.6时训练8轮测试数据的准确率达到0.89
tf.flags.DEFINE_integer("num_epochs", 10, "训练的轮数")
tf.flags.DEFINE_float("init_learning_rate", 0.001, "初始学习率")
FLAGS = tf.flags.FLAGS
x=tf.placeholder("float",[None,FLAGS.sentence_length,FLAGS.vector_size]) #输入的x变量是三维的第一维是批次大小，第二，第三维组成一个句子
y=tf.placeholder("float",[None,FLAGS.num_classes])
inputs=tf.unstack(x,axis=1)
# inputs = tf.transpose(inputs, [1,0,2])
# # 转换成(sequence_length * batch_size, rnn_size)
# inputs = tf.reshape(inputs, [-1, FLAGS.n_hidden])
# # 转换成list,里面的每个元素是(batch_size, rnn_size)
# inputs = tf.split(inputs, FLAGS.sentence_length, 0)
#静态LSTM网络的构建
lstm_fw_cell=tf.contrib.rnn.LSTMCell(FLAGS.n_hidden)
lstm_bw_cell=tf.contrib.rnn.LSTMCell(FLAGS.n_hidden)
# gru_cell=tf.nn.rnn_cell.DropoutWrapper(gru_cell,output_keep_prob=0.6)
outputs,_,_=tf.nn.static_bidirectional_rnn(lstm_fw_cell,lstm_bw_cell,inputs,dtype=tf.float32)
# 定义attention layer
attention_size = FLAGS.attn_size
#属于soft-Attention即为对于LSTM的输出层所有节点都计算attention
with tf.name_scope('attention'), tf.variable_scope('attention'):
    attention_w = tf.Variable(tf.truncated_normal([2*FLAGS.n_hidden, attention_size], stddev=0.1), name='attention_w')
    attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
    u_list = []
    #以下部分为相似度的计算，对每个时间序列计算相似度
    for t in xrange(FLAGS.sentence_length):
        # print(tf.shape(outputs[t]))
        u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
        u_list.append(u_t)
    u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
    #每个时间序列注意力权值的计算
    attn_z = []
    for t in xrange(FLAGS.sentence_length):
        z_t = tf.matmul(u_list[t], u_w)
        attn_z.append(z_t)
    # transform to batch_size * sequence_length
    attn_zconcat = tf.concat(attn_z, axis=1)
    alpha = tf.nn.softmax(attn_zconcat)
    # 和注意力权值相乘得到注意力值
    # transform to sequence_length * batch_size * 1 , same rank as outputs
    alpha_trans = tf.reshape(tf.transpose(alpha, [1,0]), [FLAGS.sentence_length, -1, 1])
    #输出和权值相乘，使得重要的特征更加突出，不重要的减小其影响
    final_output = tf.reduce_sum(outputs * alpha_trans, 0)

print(final_output.shape)
#更新后的输出在通过一个全连接层得到类别输出
# outputs shape: (sequence_length, batch_size, 2*rnn_size)
fc_w = tf.Variable(tf.truncated_normal([2*FLAGS.n_hidden, FLAGS.num_classes], stddev=0.1), name='fc_w')
fc_b = tf.Variable(tf.zeros([FLAGS.num_classes]), name='fc_b')

#self.final_output = outputs[-1]

# 用于分类任务, outputs取最终一个时刻的输出
pred = tf.matmul(final_output, fc_w) + fc_b


# outputs=tf.transpose(outputs,[1,0,2])
# pred=tf.contrib.layers.fully_connected(outputs[-1],FLAGS.num_classes,activation_fn=None)

#用精度评估模型
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#计算损失值选择优化器
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))

#反向传播优化函数
optimizer=tf.train.AdamOptimizer(learning_rate=FLAGS.init_learning_rate).minimize(cost)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    _,labels = weibo_data_helper.get_train_data()
    labels = np.array(labels)
    with open('pkl_model/data.pkl', 'rb') as fr:
        vec_list = pickle.load(fr)
    vec_list=np.array(vec_list)
    test_vec_list=vec_list[-2000:]
    test_labels=labels[-2000:]
    vec_list=vec_list[:-2000]
    labels=labels[:-2000]
#     #批量获得数据
    num_inter=int(len(labels)/FLAGS.batch_size)
    for epoch in range(FLAGS.num_epochs):
        for i in range(num_inter):
            start=i*FLAGS.batch_size
            end=(i+1)*FLAGS.batch_size
            feed_dict = {x: vec_list[start:end], y: labels[start:end]}
            sess.run(optimizer, feed_dict=feed_dict)
            if i%20==0:
                train_accuracy=accuracy.eval(feed_dict={x:vec_list[start:end],y:labels[start:end]})
                print("Epoch %d:Step %d accuracy is %f" % (epoch,i,train_accuracy))
#     #模型训练完之后进行模型的保存
    test_accuracy = accuracy.eval(feed_dict={x: test_vec_list, y: test_labels})
    print("test accuracy is %f" % test_accuracy)