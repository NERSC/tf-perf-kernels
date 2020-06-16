## a rnn1d kernel using tensorflow

import os
import warnings
import tensorflow as tf
import numpy as np
import argparse
import time
try:
    import pycuda.autoinit
    import pycuda as pyc
    have_pycuda=True
except:
    print("pycuda not installed")
    have_pycuda=False

warnings.simplefilter('ignore')
#tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}



def rnn1d(input_data, weights, cell_type, n_neurons, dtype):
    if cell_type == 'rnn':
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    elif cell_type == 'lstm':
        basic_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
    elif cell_type == 'gru':
        basic_cell = tf.contrib.rnn.GRUCell(num_units=n_neurons)
    else:
        raise Exception("cell_type could only be: rnn, lstm or gru!")

    outputs, states = tf.nn.dynamic_rnn(basic_cell, input_data, dtype=dtype)

    return tf.matmul(outputs[-1], weights['out'])
    #return outputs, states
    

def main(input_tensor_shape, cell_type, n_neurons, dtype, n_iter, n_warm, compute_type, enable_xla, agg_placement):




    #num_hidden = input_tensor_shape[1] # hidden layer num of features
    num_hidden = n_neurons
    #num_classes = n_neurons # MNIST total classes (0-9 digits)



    weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, 1]))
}
    #biases = {
    #'out': tf.Variable(tf.random_normal([num_classes]))
#}

    
    if dtype == 'float16':
        tensor_type=tf.float16
    elif dtype == 'float32':
        tensor_type=tf.float32
    else:
        raise Exception('data type can only be float16 or float32')
    
    if enable_xla:
        gpu_dev = "/device:XLA_GPU:0"
        cpu_dev = "/device:XLA_CPU:0"
    else:
        gpu_dev = "/device:GPU:0"
        cpu_dev = "/device:CPU:0"
        
    if agg_placement:
        agg_dev = cpu_dev
    else:
        agg_dev = gpu_dev

    with tf.device(agg_dev):
        #input tensor
        input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=dtype) 
        
    with tf.device(gpu_dev):
        
        #create network
        #output_result, states_cur = rnn1d(input_image, cell_type, n_neurons, tensor_type) 
        output_result = rnn1d(input_image, weights, cell_type, n_neurons, tensor_type)
        final_result = tf.reduce_sum(output_result)
        
        #init ops
        init_op = tf.initializers.global_variables()
        
    #resul ops
    if compute_type=="forward":
        with tf.device(gpu_dev):
            exec_op = final_result
    elif compute_type=="backward":
        with tf.device(gpu_dev):
            opt = tf.train.GradientDescentOptimizer(0.5)
            exec_op = opt.compute_gradients(final_result)
    elif compute_type=="calibrate":
        with tf.device(gpu_dev):
            exec_op = input_image
    else:
        raise ValueError("Error, compute_type should be either forward or backward or calibrate")
   
    #start session
    sess_config=tf.ConfigProto(allow_soft_placement=False,log_device_placement=True)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        sess.run(init_op)
        
        print("warming up for {} steps".format(n_warm))
        for i in range(n_warm):
            result = sess.run(exec_op)
        print("done")
        
        print("running for {} steps".format(n_iter))
        start = time.time()
        if have_pycuda:
            pyc.driver.start_profiler()
        for i in range(n_iter):
            result = sess.run(exec_op)
        if have_pycuda:
            pyc.driver.stop_profiler()
        end = time.time()
        print("done")
        
    duration = end-start
    print('duration {:.2f} seconds, {:.2f} seconds/call'.format(duration, duration/float(n_iter)))



if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--input_tensor_shape', type=int, default=[10,32,4], help='the shape of the input tensor')
    AP.add_argument('--cell_type', type=str, default='lstm', help='the rnn cell type\
')
    AP.add_argument('--n_neurons', type=int, default=20, help='number of neurons for\
 the layer')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="backward", help='forward or backward pass')
    AP.add_argument('--enable_xla', action="store_true", help="enable XLA support")
    AP.add_argument('--aggressive_placement', action="store_true", help='if enabled, place everything which is not convolution on the CPU')
    parsed = AP.parse_args()
    
    #print args
    for arg in vars(parsed):
        print(arg, ":", getattr(parsed, arg))
        
    
    main(input_tensor_shape=parsed.input_tensor_shape,
         cell_type=parsed.cell_type,
         n_neurons=parsed.n_neurons,
         dtype=parsed.dtype,
         n_iter=parsed.num_iterations,
         n_warm=parsed.num_warmups,
         compute_type=parsed.compute_type,
         enable_xla=parsed.enable_xla,
         agg_placement=parsed.aggressive_placement)
    
    

