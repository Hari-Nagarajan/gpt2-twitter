import tensorflow as tf

print(tf.__version__)
print(tf.config.experimental.list_physical_devices('GPU'))
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

import os
import time

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

np.random.seed(2020)


def conv_block(x, kernel_size=3):
    # Define some part of graph here

    bs, h, w, c = x.shape
    in_channels = c
    out_channels = c

    with tf.compat.v1.variable_scope('var_scope'):
        w_0 = tf.compat.v1.get_variable('w_0', [kernel_size, kernel_size, in_channels, out_channels],
                                        initializer=tf.keras.initializers.glorot_normal())
        x = tf.nn.conv2d(x, w_0, [1, 1, 1, 1], 'SAME')

    return x


def get_data_batch(spatial_size, n_channels):
    bs = 1
    h = spatial_size
    w = spatial_size
    c = n_channels

    x_np = np.random.rand(bs, h, w, c)
    x_np = x_np.astype(np.float32)
    # print('x_np.shape', x_np.shape)

    return x_np


def run_graph_part(f_name, spatial_size, n_channels, n_iter=100):
    print('=' * 60)
    print(f_name.__name__)

    #     tf.reset_default_graph()
    tf.compat.v1.reset_default_graph()

    with tf.compat.v1.Session() as sess:
        x_tf = tf.compat.v1.placeholder(tf.float32, [1, spatial_size, spatial_size, n_channels], name='input')
        z_tf = f_name(x_tf)

        sess.run(tf.compat.v1.global_variables_initializer())

        x_np = get_data_batch(spatial_size, n_channels)

        start_time = time.time()

        for _ in range(n_iter):
            z_np = sess.run(fetches=[z_tf], feed_dict={x_tf: x_np})[0]
        avr_time = (time.time() - start_time) / n_iter

        print('z_np.shape', z_np.shape)
        print('avr_time', round(avr_time, 3))

        n_total_params = 0

        for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='var_scope'):
            n_total_params += np.prod(v.get_shape().as_list())

        print('Number of parameters:', format(n_total_params, ',d'))

        # USING TENSORFLOW BENCHMARK
        benchmark = tf.test.Benchmark()
        results = benchmark.run_op_benchmark(sess=sess, op_or_tensor=z_tf,
                                             feed_dict={x_tf: x_np}, burn_iters=2, min_iters=n_iter,
                                             store_memory_usage=False, name='example')

        return results


if __name__ == '__main__':
    results = run_graph_part(conv_block, spatial_size=512, n_channels=32, n_iter=100)

