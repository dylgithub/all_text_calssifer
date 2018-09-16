#encoding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("/path/to/MNIST_data/",one_hot=True)
learning_rate=0.01
n_hidden_1=256
n_hidden_2=128
n_input=784
x=tf.placeholder("float",[None,n_input])
y=x
weights={
    "encoder_h1":tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    "encoder_h2":tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    "decoder_h1":tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1])),
    "decoder_h2":tf.Variable(tf.random_normal([n_hidden_1,n_input]))
}
biases={
    "encoder_b1":tf.Variable(tf.zeros([n_hidden_1])),
    "encoder_b2":tf.Variable(tf.zeros([n_hidden_2])),
    "decoder_b1":tf.Variable(tf.zeros([n_hidden_1])),
    "decoder_b2":tf.Variable(tf.zeros([n_input])),
}
def encoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))
    return layer_2
def decoder(x):
    layer_1=tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    layer_2=tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
    return layer_2
#输出的节点
encoder_out=encoder(x)
pred=decoder(encoder_out)
#计算损失值
cost=tf.reduce_mean(tf.pow(y-pred,2))
optimizer=tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

#开始训练
training_epochs=10 #迭代的总次数
batch_size=256
display_step=5

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    total_batch=int(mnist.train.num_examples/batch_size)
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #这里的aa和aa2的值相同，但是和后面的bb值不同，因为通过模型训练，为了使得解码
    #之后的值和原值更加接近，对损失函数进行了优化使得权重系数w发生了变化所以对于
    #第一层的输出也发生了变化，训练后的压缩特征才更能代表原来的特征
    aa=sess.run(encoder_out,feed_dict={x:batch_xs})
    print(np.shape(aa))
    # aa2=sess.run(encoder_out,feed_dict={x:batch_xs})
    # print(aa2)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            _,c=sess.run([optimizer,cost],feed_dict={x:batch_xs})
        if epoch%display_step==0:
            print("Epoch:",'%04d' % (epoch+1),"cost=","{:.9f}".format(c))
    print("训练完成")
    bb = sess.run(encoder_out, feed_dict={x: batch_xs})
    print(np.shape(bb))
    # #计算准确率
    correct_prediction=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print(accuracy.eval({x:mnist.test.images,y:mnist.test.images}))