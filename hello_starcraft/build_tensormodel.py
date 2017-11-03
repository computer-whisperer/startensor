#!/usr/bin/env python

import tensorflow as tf
import numpy as np

batch_size = 128
test_size = 256


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Feature layers used:
# screen_power (bit)
# screen_unit_type (int32)
# screen_selected (bit)
# screen_hit_point_ratio (uint8)
# screen_player_relative (uint8)
# minimap_camera (bit)
# minimap_player_relative (uint8)
# minimap_selected (bit)
#
# 8 input feature layers of size 32x32

# minerals
# vespene
# supply built
# supply available
# score
# frame
#
# 6 single-neuron inputs

# 3 actions for all units, actions for all structures
# 15 actions for unit and structure abilities
# 1 output for camera movement
# 1 output for unit selection
# 1 output for actions
# 1 output for minimap vs screen
# 2 outputs for x y selection
#
# 24 total output neurons

input_feature_layers = tf.placeholder("float", [None, 32, 32, 8])
input_data = tf.placeholder("float", [None, 6])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

c1_w = init_weights([3, 3, 8, 32])       # 3x3 conv, 8 inputs, 32 outputs
c1a = tf.nn.relu(tf.nn.conv2d(input_feature_layers, c1_w, strides=[1, 1, 1, 1], padding='SAME'))
c1b = tf.nn.max_pool(c1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
c1c = tf.nn.dropout(c1b, p_keep_conv)

c2_w = init_weights([3, 3, 32, 64])       # 3x3 conv, 32 inputs, 64 outputs
c2a = tf.nn.relu(tf.nn.conv2d(c1c, c2_w, strides=[1, 1, 1, 1], padding='SAME'))
c2b = tf.nn.max_pool(c2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
c2c = tf.nn.dropout(c2b, p_keep_conv)

c3_w = init_weights([3, 3, 64, 128])       # 3x3 conv, 64 inputs, 128 outputs
c3a = tf.nn.relu(tf.nn.conv2d(c2c, c3_w, strides=[1, 1, 1, 1], padding='SAME'))
c3b = tf.nn.max_pool(c3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
c3c = tf.nn.dropout(c3b, p_keep_conv)

l0 = tf.reshape(c3c, [-1, 8*8*128])

l1_w = init_weights([8*8*128, 1024])
l1a = tf.nn.relu(tf.matmul(l0, l1_w))
l1b = tf.nn.dropout(l1a, p_keep_hidden)

lout_w = init_weights([1024, 24])
out_action = tf.matmul(l1b, lout_w)

example_action = tf.placeholder("float", [None, 24])
cost = tf.square(tf.subtract(out_action, example_action))

train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(out_action, 1)

export_dir = "hello_starcraft/models/main.0.0.1"

#saver = tf.train.Saver(tf.trainable_variables())
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

# Launch the graph in a session
with tf.Session(graph=tf.Graph()) as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    builder.add_meta_graph_and_variables(sess, ["training"])
    #
    # save_path = saver.save(sess, "hello_starcraft/models/main.0.0.0")
    # print("Saved stuff to {}".format(save_path))
    # for i in range(100):
    #     training_batch = zip(range(0, len(trX), batch_size),
    #                          range(batch_size, len(trX)+1, batch_size))
    #     for start, end in training_batch:
    #         sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end],
    #                                       p_keep_conv: 0.8, p_keep_hidden: 0.5})
    #
    #     test_indices = np.arange(len(teX)) # Get A Test Batch
    #     np.random.shuffle(test_indices)
    #     test_indices = test_indices[0:test_size]
    #
    #     print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==
    #                      sess.run(predict_op, feed_dict={X: teX[test_indices],
    #                                                      p_keep_conv: 1.0,
    #                                                      p_keep_hidden: 1.0})))
builder.save()