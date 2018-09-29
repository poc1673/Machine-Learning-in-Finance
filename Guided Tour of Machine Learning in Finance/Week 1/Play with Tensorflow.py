import tensorflow as tf

# Test 1:  Implement a simple calculator to output values from an equation


x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = (4-x)**2 + 2*(y-x**2)**2

with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = f.eval()

print(result)

# Test 2:  Implement root finding on the equation:
n_epochs = 1000
learning_rate = 0.01
x = tf.Variable(10.0, trainable=True)
y = tf.Variable(3, trainable=True)
f = (4-x)**2 + 2*(y-x**2)**2
loss = tf.abs(f)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        print(i)
        res = sess.run(opt)



import tensorflow as tf

x = tf.Variable(10.0, trainable=True)
y = tf.Variable(4.0, trainable=True)
f_x = (4-x)**2 + 2*(y-x**2)**2

loss = tf.abs(f_x)
opt = tf.train.AdamOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while(loss.eval()>.1):
        print(sess.run([x,y,loss]))
        sess.run(opt)
        
        
        
        
        

# Test 3: Implement linear regression using gradient descent.



# Test 4:  Implement regularized linear regression with gradient descent:





# Test 5: Implement logistic regression using log-likelihood.





# Test 6:  Implement Tobit regression 