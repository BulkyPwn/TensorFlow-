import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用 numpy 生成 200 个 随机点
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#定义神经网络中间层 1->10->1
Weights_L1 = tf.Variable(tf.random.normal([1,10]))#权重
biases_L1 = tf.Variable(tf.zeros([1,10]))#偏置 取决于下一层神经元个数
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + biases_L1#信号总和
L1 = tf.nn.tanh(Wx_plus_b_L1)#双曲正切函数

#定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#定义一个 梯度下降法 来训练的 优化器
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#最小化代价函数
#train = optimizer.minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(20000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})
    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,"r-",lw=5)
    plt.show()
