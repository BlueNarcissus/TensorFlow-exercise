import tensorflow as tf

""" (1) constant """
# Build Computational Graph
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # tf.float32 implicitly
print(node1, node2)

# Run enough of Computational Graph
sess = tf.Session() # Session object
print(sess.run([node1, node2])) # Run method

node3 = tf.add(node1, node2) # create a new node, operation
print(node3)
print(sess.run(node3))

""" (2) placeholder: provide a value later """
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b   #adder_node = tf.add(a,b)
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

add_triple_node = adder_node*3
print(sess.run(add_triple_node, feed_dict={a:3, b:4.5}))

""" (3) Variable: add trainable parameters to a graph """
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W*x+b

# Initialize global variables
init = tf.global_variables_initializer()
sess.run(init)

# loss function
loss = tf.reduce_sum(tf.square(linear_model-y))
print(sess.run(loss, feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]}))

# assign optimal paramters to the model
fixW = tf.assgin(W, [-1.0])
fixb = tf.assgin(b, [1.0])
sess.run([fixW, fixb])
print(sess.run(loss, feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]}))


""" train a model """
optimizer = tf.train.GradientDescentOptimizer(0.01) # learning_rate=0.01
train = optimizer.minimize(loss)
sess.run(init)
for i in range(1000):
    sess.run(train, feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W,b,loss], feed_dict={x:[1,2,3,4], y:[0,-1,-2,-3]}))

