import argparse
import sys
import tensorflow as tf

from tensorflow.python.platform import app
from tensorflow.python.summary import summary

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(
        sess, [tf.saved_model.tag_constants.SERVING], './')

    pb_visual_writer = summary.FileWriter('/tmp/tensorflow_logdir')
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format('/tmp/tensorflow_logdir'))