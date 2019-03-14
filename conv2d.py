## a conv2d kernel using tensorflow

import os
import warnings
import tensorflow as tf
import numpy as np
import argparse
import time

warnings.simplefilter('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}


#N = 10
#C = 3
#H = 128
#W = 128

#input_tensor_shape = [10,128,128,3]  # NHWC
#kernel_shape = [5,5,3,10]
#stride_ = [1,2,2,1]

def conv2d(input_data, data_format, kernel_shape, stride_, dtype):
    weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.03, dtype=dtype), dtype=dtype)
    if data_format == "NCHW":
        input_data = tf.transpose(input_data, [0,3,1,2])
        strides = [1,1,stride_, stride_]
    else:
        strides = [1,stride_,stride_,1]
    output_data = tf.nn.conv2d(input_data, weights, strides=strides, padding='SAME', data_format=data_format)
    return output_data

def main(input_tensor_shape, data_format, kernel_shape, stride_, dtype, n_iter, n_warm, compute_type):

    with tf.device('/device:gpu:0'):

        if dtype == 'float16':
            tensor_type=tf.float16
        elif dtype == 'float32':
            tensor_type=tf.float32
        else:
            raise Exception('data type can only be float16 or float32')

        #shapes
        #input_tensor_shape = [int(i) for i in input_tensor_shape.split(',')]
        #kernel_shape = [int(i) for i in kernel_shape.split(',')]
        
        #create network
        input_image = tf.random_uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=dtype) #np.ndarray(input_tensor_shape, dtype=dtype)
        output_result = conv2d(input_image, data_format, kernel_shape, stride_, tensor_type) 
        
        #init ops
        init_op = tf.global_variables_initializer()
        
        #resul ops
        if compute_type=="forward":
            exec_op = tf.identity(output_result)
        elif compute_type=="backward":
            peudo_loss = tf.nn.l2_loss(output_result)
            opt = tf.train.GradientDescentOptimizer(0.5)
            exec_op = opt.compute_gradients(pseudo_loss)
        else:
            raise ValueError("Error, compute_type should be either forward or backward")
   
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
        sess.run(init_op)
        
        print("warming up")
        for i in range(n_warm):
            result = sess.run(exec_op)
        print("done")
        
        print("starting")
        start = time.time()
        for i in range(n_iter):
            result = sess.run(exec_op)
        end = time.time()
        print("done")
        
    duration = end-start
    print('duration {:.2f} seconds, {:.2f} seconds/call'.format(duration, duration/float(n_iter)))



if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--input_tensor_shape', type=int, nargs='+', help='the shape of the input tensor. Note that it depends on data_format (default NHWC)')
    AP.add_argument('--data_format', type=str, default='NHWC', help='choose either channels_last or channels_first')
    AP.add_argument('--kernel_shape', type=int, nargs='+', default=[5,5,3,32], help='the shape of the conv kernel [filter_height, filter_width, in_channels, out_channels]')
    AP.add_argument('--stride', type=int, default=1, help='the stride')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="forward", help='forward or backward pass')
    parsed = AP.parse_args()
    
    #print args
    for arg in vars(parsed):
        print(arg, ":", getattr(parsed, arg))
        
    
    main(input_tensor_shape=parsed.input_tensor_shape,
         data_format=parsed.data_format,
         kernel_shape=parsed.kernel_shape,
         stride_=parsed.stride,
         dtype=parsed.dtype,
         n_iter=parsed.num_iterations,
         n_warm=parsed.num_warmups,
         compute_type=parsed.compute_type)
    
    

