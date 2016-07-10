# this is one very good result taken from this post
# https://discussions.udacity.com/t/assignment-4-problem-2/46525/24
# by user endri.deliu the 10. Feb 2016
"""
I am using 1024x1024x305x75 with 95k steps, learning rate decay and
dropout. I get consistently 92.8-92.9% on validation set and 97.3% on
the test set. I am using 350k images on the training set, if you use
the full set of 450k its likely the results will improve. The key in
getting these results was to initialize the weights properly (used the
formula stddev=sqrt(2/n)) and stacked more layers. I am pretty sure
you can get better than my result if you stack even more layers and
maybe reduce the width of the network to stay within similar
computational bounds, plus you can also add L2 regularization in
addition to dropout and train longer. With convolutions you may not
get this far and they may be overkill for OCR with images of size
28x28. But I may be wrong here, it would be nice to see somebody
approaching the same results with convolutions. Maybe training longer
would do the trick.

"""
batch_size = 128
hidden_layer1_size = 1024
hidden_layer2_size = 305
hidden_lastlayer_size = 75

use_multilayers = True

regularization_meta=0.03


graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

  # Variables.
  keep_prob = tf.placeholder(tf.float32)

  weights_layer1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_layer1_size], stddev=0.0517))
  biases_layer1 = tf.Variable(tf.zeros([hidden_layer1_size]))

  if use_multilayers:
    weights_layer2 = tf.Variable(
      tf.truncated_normal([hidden_layer1_size, hidden_layer1_size], stddev=0.0441))
    biases_layer2 = tf.Variable(tf.zeros([hidden_layer1_size]))

    weights_layer3 = tf.Variable(
      tf.truncated_normal([hidden_layer1_size, hidden_layer2_size], stddev=0.0441))
    biases_layer3 = tf.Variable(tf.zeros([hidden_layer2_size]))

    weights_layer4 = tf.Variable(
      tf.truncated_normal([hidden_layer2_size, hidden_lastlayer_size], stddev=0.0809))
    biases_layer4 = tf.Variable(tf.zeros([hidden_lastlayer_size]))


  weights = tf.Variable(
    tf.truncated_normal([hidden_lastlayer_size if use_multilayers else hidden_layer1_size, num_labels], stddev=0.1632))
  biases = tf.Variable(tf.zeros([num_labels]))


  # get the NN models
  def getNN4Layer(dSet, use_dropout):
    input_to_layer1 = tf.matmul(dSet, weights_layer1) + biases_layer1
    hidden_layer1_output = tf.nn.relu(input_to_layer1)


    logits_hidden1 = None
    if use_dropout:
       dropout_hidden1 = tf.nn.dropout(hidden_layer1_output, keep_prob)
       logits_hidden1 = tf.matmul(dropout_hidden1, weights_layer2) + biases_layer2
    else:
      logits_hidden1 = tf.matmul(hidden_layer1_output, weights_layer2) + biases_layer2

    hidden_layer2_output = tf.nn.relu(logits_hidden1)

    logits_hidden2 = None
    if use_dropout:
       dropout_hidden2 = tf.nn.dropout(hidden_layer2_output, keep_prob)
       logits_hidden2 = tf.matmul(dropout_hidden2, weights_layer3) + biases_layer3
    else:
      logits_hidden2 = tf.matmul(hidden_layer2_output, weights_layer3) + biases_layer3


    hidden_layer3_output = tf.nn.relu(logits_hidden2)
    logits_hidden3 = None
    if use_dropout:
       dropout_hidden3 = tf.nn.dropout(hidden_layer3_output, keep_prob)
       logits_hidden3 = tf.matmul(dropout_hidden3, weights_layer4) + biases_layer4
    else:
      logits_hidden3 = tf.matmul(hidden_layer3_output, weights_layer4) + biases_layer4


    hidden_layer4_output = tf.nn.relu(logits_hidden3)
    logits = None
    if use_dropout:
       dropout_hidden4 = tf.nn.dropout(hidden_layer4_output, keep_prob)
       logits = tf.matmul(dropout_hidden4, weights) + biases
    else:
      logits = tf.matmul(hidden_layer4_output, weights) + biases

    return logits

  # get the NN models
  def getNN1Layer(dSet, use_dropout, w1, b1, w, b):
    input_to_layer1 = tf.matmul(dSet, w1) + b1
    hidden_layer1_output = tf.nn.relu(input_to_layer1)

    logits = None
    if use_dropout:
       dropout_hidden1 = tf.nn.dropout(hidden_layer1_output, keep_prob)
       logits = tf.matmul(dropout_hidden1, w) + b
    else:
      logits = tf.matmul(hidden_layer1_output, w) + b

    return logits



  # Training computation.
  logits = getNN4Layer(tf_train_dataset, True)
  logits_valid = getNN4Layer(tf_valid_dataset, False)
  logits_test = getNN4Layer(tf_test_dataset, False)


  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  #loss_l2 = loss + (regularization_meta * (tf.nn.l2_loss(weights)))

  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(0.3, global_step, 3500, 0.86, staircase=True)


  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(logits_valid)
  test_prediction = tf.nn.softmax(logits_test)



num_steps = 95001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print "Initialized"
  for step in xrange(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]

    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:0.75}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print "Minibatch loss at step", step, ":", l
      print "Minibatch accuracy: %.1f%%" % accuracy(train_prediction.eval(feed_dict={tf_train_dataset : batch_data, tf_train_labels : batch_labels, keep_prob:1.0}), batch_labels)
      print "Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(feed_dict={keep_prob:1.0}), valid_labels)
  print "Test accuracy: %.1f%%" % accuracy(test_prediction.eval(feed_dict={keep_prob:1.0}), test_labels)
