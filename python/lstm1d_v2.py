## a conv2d kernel using tensorflow

import os
import warnings
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#import tensorflow as tf
#from tensorflow.python.eager import profiler as tfprof
import numpy as np
import argparse
import time
try:
    import pycuda.autoinit
    import pycuda as pyc
    have_pycuda=True
    print("pycuda enabled")
except:
    print("pycuda not installed")
    have_pycuda=False

#warnings.simplefilter('ignore')
#tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
print("Eager execution: {}".format(tf.executing_eagerly()))

#    #set gpu to allow mem growth
#    gpus = tf.config.experimental.list_physical_devices('GPU')
#    if gpus:
#        try:
#            # Currently, memory growth needs to be the same across GPUs
#            for gpu in gpus:
#                tf.config.experimental.set_memory_growth(gpu, True)
#            #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#        except RuntimeError as e:
#            # Memory growth must be set before GPUs have been initialized
#            print(e)


def rnn1d(input_data, cell_type, n_neurons, dtype):
    if cell_type == 'rnn':
        basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
    elif cell_type == 'lstm':
        basic_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_neurons)
    elif cell_type == 'gru':
        basic_cell = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
    else:
        raise Exception("cell_type could only be: rnn, lstm or gru!")

    outputs, states = tf.nn.dynamic_rnn(basic_cell, input_data, dtype=dtype)

    return outputs, states



#calibration measurement
def run_calibrate(input_tensor_shape, cell_type, n_neurons, tensor_type):
    #define op
    #run the stuff
    input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=tensor_type)
    _ = input_image.numpy()


#forward
def run_forward(input_tensor_shape, cell_type, n_neurons, tensor_type):
    #define op
    input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=tensor_type)

    output_result, states_cur = rnn1d(input_image, cell_type, n_neurons, tensor_type)
    
    #run the stuff
    _ = output_result.numpy()


#backward
def run_backward(input_tensor_shape, cell_type, n_neurons, tensor_type):
    #define op, under tape
    input_image = tf.random.uniform(shape=input_tensor_shape, minval=0., maxval=1., dtype=tensor_type)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_image)
        output_result, states_cur = rnn1d(input_image, cell_type, n_neurons, tensor_type)
    grad_input   = tape.gradient(output_result, input_image)
    grad_weights = tape.gradient(output_result, weights)
    
    #run the stuff
    _, _ = grad_input.numpy(), grad_input.numpy()


def main(input_tensor_shape, cell_type, n_neurons, dtype, n_iter, n_warm, compute_type, enable_xla):
    
    #datatype selection
    if dtype == 'float16':
        tensor_type=tf.float16
    elif dtype == 'float32':
        tensor_type=tf.float32
    else:
        raise Exception('data type can only be float16 or float32')
    
    ##XLA or not
    if tf.test.is_gpu_available():
        device = '/xla_gpu:0' if enable_xla else '/gpu:0'
    else:
        device = '/xla_cpu:0' if enable_xla else '/cpu:0'
        
    print("Running on device {}".format(device))
    #tf.config.experimental.set_memory_growth(device, True)
    #tf.config.gpu.set_per_process_memory_growth(True)

    # select commpute type
    if compute_type == "forward":
        compfunc = run_forward
    elif compute_type == "backward":
        compfunc = run_backward
    elif compute_type == "calibrate":
        compfunc = run_calibrate
    else:
        raise ValueError("Error, compute_type should be either forward or backward or calibrate")
    
    #we might need that
#    with tf.device(device):
#        weights = tf.Variable(tf.random.truncated_normal(kernel_shape, stddev=0.03, dtype=dtype), dtype=dtype)
    
    #start session
    print("warming up for {} steps".format(n_warm))
    with tf.device(device):
        for i in range(n_warm):
            compfunc(input_tensor_shape, cell_type, n_neurons, tensor_type)
    print("done")
        
    print("running for {} steps".format(n_iter))
    start = time.time()
    #start profiling
    if have_pycuda:
        pyc.driver.start_profiler()
    with tf.device(device):
        for i in range(n_iter):
            compfunc(input_tensor_shape, cell_type, n_neurons, tensor_type)

    #stop profiling
    if have_pycuda:
        pyc.driver.stop_profiler()
    end = time.time()
    print("done")
    
    duration = end-start
    print('duration {:.2f} seconds, {:.2f} seconds/call'.format(duration, duration/float(n_iter)))



if __name__ == '__main__':
    AP = argparse.ArgumentParser()
    AP.add_argument('--input_tensor_shape', type=int, default=[10,32,32], help='the shape of the input tensor')
    AP.add_argument('--cell_type', type=str, default='lstm', help='the rnn cell type\
')
    AP.add_argument('--n_neurons', type=int, default=50, help='number of neurons for\
 the layer')
    AP.add_argument('--dtype', type=str, default='float32', help='the data type')
    AP.add_argument('--num_iterations', type=int, default=100, help='the number of iterations')
    AP.add_argument('--num_warmups', type=int, default=10, help='number of warmup steps')
    AP.add_argument('--compute_type', type=str, default="forward", help='forward or backward pass')
    AP.add_argument('--enable_xla', action="store_true", help="enable XLA support")
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
         enable_xla=parsed.enable_xla)
    
    

