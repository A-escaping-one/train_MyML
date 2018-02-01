#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#增加节点
def add_layers(inputs,in_size,out_size,activtion_function=None):
    #设置权重 y = wx+b 的 w
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    #设置偏差 y = wx+b 的 b
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    #写出方程y = wx+b
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if activtion_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activtion_function(Wx_plus_b)
    return outputs
#初始化训练数据集 
x_data = np.linspace(-1,1,300)[:,np.newaxis]
#设置噪点
noise = np.random.normal(0,0.05,x_data.shape)
#y = x的平方 + 噪点 （增加噪点是为了数据比较接近现实）
y_data = np.square(x_data) + noise

#设置输入的值
xs = tf.placeholder(tf.float32,[None,1])
ys = tf.placeholder(tf.float32,[None,1])

#增加节点1个输入，10个输出，激活函数选择relu函数
l1 = add_layers(xs,1,10,activtion_function=tf.nn.relu)
#增加系欸但 10个输入，1个输出，激活函数不用 ，加上上面那个节点，形成两层神经层
predition = add_layers(l1,10,1,activtion_function=None)
#损失函数 求出实际值和训练值的差值，然后使用梯度下降进行修正（给出函数，tensorflow会自行求导进行梯度下降）
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-predition),reduction_indices=[1]))#reduction_indices=[1]按行求和
#学习步数，设置学习率0.1，把损失函数放入（设置这个这么低的学习率，个人理解，是怕错过最优值）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#初始化所有参数，必须加
init = tf.initialize_all_variables()
#画
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()
#拿取session，tensorflow很重要东西，拿取训练过程的东西都要用这个
sess = tf.Session()
sess.run(init)
for i in range(1000):
    #开始训练，喂入数据feed_dict
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        # print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        #得到训练后的数据
        prediction_value=sess.run(predition,feed_dict={xs:x_data})
        #将数据可视化
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)
        plt.pause(0.1)
