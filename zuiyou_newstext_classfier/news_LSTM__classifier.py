#encoding=utf-8
import tensorflow as tf
import numpy as np
import time
from zuiyou_newstext_classfier import weibo_data_helper
import pickle as pickle
#0.819
# 模型的超参数
tf.flags.DEFINE_integer("vector_size", 80, "每个单词向量的维度")
tf.flags.DEFINE_integer("sentence_length", 200, "设定的句子最大长度")
tf.flags.DEFINE_integer("n_hidden", 100, "隐藏层细胞的个数")
tf.flags.DEFINE_integer("num_classes", 10, "类别种类数")
# 训练参数
tf.flags.DEFINE_integer("batch_size", 64, "每个批次的大小")
#output_keep_prob=0.6时训练8轮测试数据的准确率达到0.89
tf.flags.DEFINE_integer("num_epochs", 10, "训练的轮数")
tf.flags.DEFINE_float("init_learning_rate", 0.001, "初始学习率")
FLAGS = tf.flags.FLAGS
x=tf.placeholder("float",[None,FLAGS.sentence_length,FLAGS.vector_size]) #输入的x变量是三维的第一维是批次大小，第二，第三维组成一个句子
y=tf.placeholder("float",[None,FLAGS.num_classes])
#动态GRU网络的构建
gru_cell=tf.contrib.rnn.LSTMCell(FLAGS.n_hidden)
gru_cell=tf.nn.rnn_cell.DropoutWrapper(gru_cell,output_keep_prob=0.6)
outputs,_=tf.nn.dynamic_rnn(gru_cell,x,dtype=tf.float32)
#转换为时间批次优先
outputs=tf.transpose(outputs,[1,0,2])
#outputs[-1]取最后时刻的输出作为本文表示
# print(tf.shape(outputs))
pred=tf.contrib.layers.fully_connected(outputs[-1],FLAGS.num_classes,activation_fn=None)

#用精度评估模型
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

saver = tf.train.Saver()  # 生成用于模型保存与加载的saver
#加载模型对新数据进行预测
# with tf.Session() as sess2:
#     tf.global_variables_initializer().run()
#     saver.restore(sess2, "tf_lstm_model/linermodel.cpkt")
#     vec_list, lables = weibo_data_helper.get_vec_data(FLAGS.sentence_length, FLAGS.vector_size)
#     vec_list = np.array(vec_list)
#     labels = np.array(lables)
#     test_vec_list = vec_list[-5000:]
#     test_labels = labels[-5000:]
#     test_accuracy = accuracy.eval(feed_dict={x: test_vec_list, y: test_labels})
#     print("test accuracy is %f" % test_accuracy)





#计算损失值选择优化器
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
# tf.summary.scalar('loss_function', cost)

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
            sess.run(optimizer,feed_dict=feed_dict)
            if i%20==0:
                train_accuracy=accuracy.eval(feed_dict={x:vec_list[start:end],y:labels[start:end]})
                print("Epoch %d:Step %d accuracy is %f" % (epoch,i,train_accuracy))
#     #模型训练完之后进行模型的保存
    saver.save(sess, "tf_lstm_model/linermodel.cpkt")
    test_accuracy = accuracy.eval(feed_dict={x: test_vec_list, y: test_labels})
    print("test accuracy is %f" % test_accuracy)
#     end_time=time.time()
#     print('总共花费：',end_time-start_time)