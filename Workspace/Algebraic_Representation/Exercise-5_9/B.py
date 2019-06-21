import input_data
import tensorflow as tf


'''

B) Training on MNIST
---------------------
Design a neural network to classify digits from the MNIST dataset. 
Start by reasoning about the number and size of hidden layers, 
and document your considerations.
Train the network on the training portion of the dataset.


Considerations:
---------------
After some research online I found the video: TensorFlow in 5 Minutes (tutorial)
(https://www.youtube.com/watch?v=2FmcHiLCwTU) 
I've desided to base my code on this.

This uses the resolution of the image input layer (28*28=784) what seems solid to me.

'''


mnist = input_data.read_data_sets("D:/Documents_D/HBO-ICT/git/HBO-ICT/Jaar4/TCTI-VKAAI-17/Workspace/Algebraic_Representation/Exercise-5_9/data", one_hot=True)
mnist = input_data.read_data_sets("D:/Documents_D/HBO-ICT/git/HBO-ICT/Jaar4/TCTI-VKAAI-17/Workspace/Algebraic_Representation/Exercise-5_9/data", one_hot=True)


learning_rate 		= 0.01
training_iteration 	= 30
batch_size 			= 100
display_step 		= 2

# TF graph input
x = tf.compat.v1.placeholder("float", [None, 784])
y = tf.compat.v1.placeholder("float", [None, 10])

# Create a model

# Set model Weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


with tf.name_scope("Wx_b") as scope:
	# Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b)

# Add summary ops to collect data
w_h = tf.compat.v1.summary.histogram("weights", W)
b_h = tf.compat.v1.summary.histogram("biases", b)


# More name scopes will clean up graph representation
with tf.name_scope("cost_function") as scope:
	# Minimize error using cross entropy
	cost_function = -tf.reduce_sum(y * tf.math.log(model))
	# Create a summary to monitor the cost function
	tf.compat.v1.summary.scalar("cost_function", cost_function)


with tf.name_scope("train") as scope:
	# Gradient descent
	optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.initialize_all_variables()

# Merge all summaries into a single operator
merged_summary = tf.compat.v1.summary.merge_all()



# Launch the graph
with tf.compat.v1.Session() as sess:
	sess.run(init)
	
	# Set the logs writer to the folder /tmp/logs
	summary_writer = tf.summary.FileWriter("D:/Documents_D/HBO-ICT/git/HBO-ICT/Jaar4/TCTI-VKAAI-17/Workspace/Algebraic_Representation/Exercise-5_9/logs", graph=sess.graph)
	
	# Training cycle
	for iteration in range(training_iteration):
		avg_cost = 0.0
		total_batch = int(mnist.train.num_examples / batch_size)
		
		# Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			# Compute the avg loss
			avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/ total_batch
			# Write logs for each iteration
			summary_str = sess.run(merged_summary, feed_dict={x: batch_xs, y: batch_ys})
			summary_writer.add_summary(summary_str, iteration * total_batch + 1)
			
		# Display logs per iteration step
		if iteration % display_step == 0:
			print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

	print("Training done")
	
	# Test the model
	predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
	print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
	


