import tensorflow as tf

# model: parameters, input and output
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x+b

# loss & optimizer
loss = tf.reduce_sum(tf.square(linear_model-y))
optimizer = tf.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# train the linear regression model
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000):
    sess.run(train, feed_dict={x:x_train, y:y_train})

# Evaluate training accuracy
print(sess.run([curr_W, curr_b, loss], feed_dict={x:x_train, y:y_train}))
