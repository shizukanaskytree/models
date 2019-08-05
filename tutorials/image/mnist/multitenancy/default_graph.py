import tensorflow as tf

g = tf.Graph(graph_priority = 1)
with g.as_default():
    c = tf.constant(5.0)
    assert c.graph is g


