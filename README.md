# tf-perf-kernels
This repository contains scripts calling kernels for TensorFlow along with profiling scripts for Cori-GPU.

## Prerequisites (Cori)
`module load python/3.7-anaconda-2019.07 cuda/10.2.89`

### [Install TensorFlow (Conda)](https://www.tensorflow.org/install/pip?lang=python3#package-location)
Create a new virtual environment `py3.7-tf2` by choosing Python 3.7:
```bash
conda create -n py3.7-tf2 pip python=3.7
```

Activate the virtual environment and install TensorFlow (Python 3.7 GPU support):
```bash
source activate py3.7-tf2
(env) pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.0.0-cp37-cp37m-manylinux2010_x86_64.whl
conda deactivate
```

### [Install PyCUDA](https://wiki.tiker.net/PyCuda/Installation/Linux)
Download PyCUDA and unpack it:
```bash
wget https://files.pythonhosted.org/packages/5e/3f/5658c38579b41866ba21ee1b5020b8225cec86fe717e4b1c5c972de0a33c/pycuda-2019.1.2.tar.gz
tar xvf pycuda-*.tar.gz
```

Configure and build within the Conda env:
```bash
cd pycuda-*
source activate py3.7-tf2
(env) python configure.py --cuda-root=${CUDA_HOME}
(env) make install -j8
conda deactivate
```
