#encoding=utf-8
import pandas as pd
import pickle as pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from gensim.models import word2vec
import tensorflow as tf
def get_train_data():
    df = pd.read_csv('data/cut_train_data.csv', encoding='utf-8')
    content_list = list(df['word'])[:10000]
    content_list=[content.strip() for content in content_list]
    label_list=list(df['label'])
    # jieba_cut_list=[jieba.lcut(content) for content in content_list]
    label_encoder=LabelEncoder()
    encoder=label_encoder.fit_transform(label_list)
    one_hot=OneHotEncoder()
    one_hot_label=one_hot.fit_transform(np.array(encoder).reshape([-1,1]))
    one_hot_label=one_hot_label.toarray()
    jieba_cut_list=[content.split(' ') for content in content_list]
    jieba_cut_list=[[word for word in jieba_cut if word!=''] for jieba_cut in jieba_cut_list]
    print('final get_train_data')
    return jieba_cut_list,one_hot_label[:10000]
#直接获得数据不在进行look_embedding
def get_pkl_model(data_num,sentence,vector_size):
    jieba_cut_list,_=get_train_data()
    word2vec_model = word2vec.Word2Vec(jieba_cut_list, min_count=1, window=2, size=80)
    vec_list=[]
    for jieba_cut in tqdm(jieba_cut_list):
        sen_vec=[]
        for i,word in enumerate(jieba_cut):
            if i>=sentence:
                break
            sen_vec.append(list(word2vec_model[word]))
        sen_len=len(jieba_cut)
        if sen_len<sentence:
            for j in range(sentence-sen_len):
                sen_vec.append([0]*vector_size)
        vec_list.append(sen_vec)
    vec_list=np.array(vec_list).reshape((data_num,sentence,vector_size))
    with open('pkl_model/data.pkl','wb') as fr:
        pickle.dump(vec_list,fr)
def get_autoencoder_pkl(data_num,sentence,vector_size):
    jieba_cut_list, _ = get_train_data()
    word2vec_model = word2vec.Word2Vec(jieba_cut_list, min_count=1, window=2, size=80)
    vec_list = []
    for jieba_cut in tqdm(jieba_cut_list):
        for i, word in enumerate(jieba_cut):
            if i >= sentence:
                break
            vec_list.append(list(word2vec_model[word]))
        sen_len = len(jieba_cut)
        if sen_len < sentence:
            for j in range(sentence - sen_len):
                vec_list.append([0] * vector_size)
    vec_list=np.array(vec_list)
    print(vec_list.shape)
    # vec_list=vec_list.reshape((-1,200,80))
    # print(vec_list.shape)
    learning_rate = 0.01
    n_hidden_1 = 40
    n_hidden_2 = 10
    n_input = 80
    x = tf.placeholder("float", [None, n_input])
    y = x
    weights = {
        "encoder_h1": tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        "encoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        "decoder_h1": tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        "decoder_h2": tf.Variable(tf.random_normal([n_hidden_1, n_input]))
    }
    biases = {
        "encoder_b1": tf.Variable(tf.zeros([n_hidden_1])),
        "encoder_b2": tf.Variable(tf.zeros([n_hidden_2])),
        "decoder_b1": tf.Variable(tf.zeros([n_hidden_1])),
        "decoder_b2": tf.Variable(tf.zeros([n_input])),
    }

    def encoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
        return layer_2

    def decoder(x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
        return layer_2

    # 输出的节点
    encoder_out = encoder(x)
    pred = decoder(encoder_out)
    # 计算损失值
    cost = tf.reduce_mean(tf.pow(y - pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 开始训练
    training_epochs = 1 # 迭代的总次数
    batch_size = 512
    display_step = 5

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        total_batch = int(len(vec_list) / batch_size)
        # 这里的aa和aa2的值相同，但是和后面的bb值不同，因为通过模型训练，为了使得解码
        # 之后的值和原值更加接近，对损失函数进行了优化使得权重系数w发生了变化所以对于
        # 第一层的输出也发生了变化，训练后的压缩特征才更能代表原来的特征
        # aa2=sess.run(encoder_out,feed_dict={x:batch_xs})
        # print(aa2)
        a_array=tf.zeros([1,80])
        for epoch in range(training_epochs):
            for i in range(total_batch):
                if (i+1)*batch_size>len(vec_list):
                    batch_xs=vec_list[i*batch_size:]
                else:
                    batch_xs = vec_list[i * batch_size:(i+1) * batch_size]
                _, c ,pred2= sess.run([optimizer, cost,pred], feed_dict={x: batch_xs})
                print(i)
                a_array=tf.concat([a_array,pred2],axis=0)
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
        print(a_array.shape)
        print("训练完成")
        # vec_list.reshape((data_num,sentence,vector_size))
        # with open('pkl_model/encoder_data.pkl', 'wb') as fr:
        #     pickle.dump(vec_list, fr)
if __name__ == '__main__':
    # get_pkl_model(10000,200,80)
    get_autoencoder_pkl(10000,200,80)
#     with open('pkl_model/data.pkl','rb') as fr:
#         X=pickle.load(fr)
#     X=np.array(X)
#     X=np.expand_dims(X,axis=-1)
#     print(X.shape)