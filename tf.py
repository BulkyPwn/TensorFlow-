import tensorflow as tf

x = tf.Variable([1,2])
a = tf.constant([3,3])
#增加一个减法op
sub = tf.subtract(x,a)
#增加一个加法op
add = tf.add(x,sub)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

#创建一个变量，初始化为0
state = tf.Variable(0,name="counter")
#创建一个op，作用是使state加1
new_value = tf.add(state,1)
#赋值op
update = tf.assign(state,new_value)
#变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))


#Fetch  同时运行多个op
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    result = sess.run([mul,add])
    print(result)

#Feed  运行时再将值传入
input1 = tf.placeholder(tf.float32)#创建占位符
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))




