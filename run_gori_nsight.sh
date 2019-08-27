#!/bin/bash
#SBATCH -J conv2d_test
#SBATCH -t 02:00:00
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive

#load modules
module unload cuda
module load cuda/10.1.243
#module load cuda/10.0.130
module load python/3.7-anaconda-2019.07

#activate env
source activate thorstendl-gori-py3-tf2
#module load tensorflow/gpu-1.13.1-py36
#module load tensorflow/gpu-2.0.0-beta-py36

#rankspernode
rankspernode=1

#openmp stuff
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
sruncmd="srun -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 80 / ${rankspernode} )) --cpu_bind=cores"


#create run dir
run_dir=$WORK/tf_cnn_kernels_2/runs/${SLURM_JOBID}
mkdir -p ${run_dir}

#copy relevant files
cp conv2d_v2.py ${run_dir}/

#variables
prec=16
batch_size=16
data_format="NHWC"

#net_params
#net_params_1="VGG-1,224x224x3,3x3x3x64,1 ResNet50-1,224x224x3,7x7x3x64,2 VGG-2,224x224x64,3x3x64x128,2 VGG-3,112x112x128,3x3x128x256,2 ResNet50-2,112x112x64,3x3x64x64,2"
#net_params_2="ResNet50-2,112x112x64,3x3x64x128,2 ResNet50-2,112x112x64,3x3x64x256,2 ResNet50-2,112x112x64,3x3x64x512,2 ResNet50-2,112x112x64,7x7x64x64,2 ResNet50-2,112x112x64,9x9x64x64,2"
#net_params_3="ResNet50-2,112x112x64,3x3x64x128,1 ResNet50-2,112x112x64,3x3x64x256,1 ResNet50-2,112x112x64,3x3x64x512,1 ResNet50-2,112x112x64,7x7x64x64,1 ResNet50-2,112x112x64,9x9x64x64,1"
#net_params_4="ResNet50-2,112x112x64,3x3x64x128,3 ResNet50-2,112x112x64,3x3x64x256,3 ResNet50-2,112x112x64,3x3x64x512,3 ResNet50-2,112x112x64,7x7x64x64,3 ResNet50-2,112x112x64,9x9x64x64,3"
#net_params="ResNet50-2,112x112x64,3x3x64x64,2 ResNet50-2,112x112x64,3x3x64x64,3"
net_params="ResNet50-2,112x112x64,3x3x64x128,2"


#step in
cd ${run_dir}

#list of metrics
#metrics=""
metrics="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum" #,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"

#iterate over metrics
for metric in ${metrics}; do
    
    #iterate over input tuples
    for input in ${net_params}; do 
        OLDIFS=$IFS; IFS=','
        set -- $input; 
        name=$1
        input_tensor_shape=${2//x/ }
        kernel_shape=${3//x/ }
        stride=${4}
        IFS=${OLDIFS}

        #iterate over FW BW
        for ctype in calibrate forward backward; do

            #get better metric name
            metricname=${metric//,/-}
    
            #assemble profiling string
            if [ "${metric}" == "time" ]; then
                profilestring="nv-nsight-cu-cli --profile-from-start off"
                #profilestring="nv-nsight-cu-cli"
            else
                profilestring="nv-nsight-cu-cli --profile-from-start off --metrics ${metric}"
                #profilestring="nv-nsight-cu-cli --metrics ${metric}"
            fi
            profilestring=${profilestring}" -f -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}.metric_${metricname}.nvvp"
	    #profilestring=${profilestring}" -f -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}.metric_${metricname}.nsight"

            #compute types
            ${sruncmd} ${profilestring} $(which python) -u ./conv2d.py \
                --dtype float${prec} \
                --data_format ${data_format} \
                --input_tensor_shape ${batch_size} ${input_tensor_shape} \
                --kernel_shape ${kernel_shape} \
                --stride ${stride} \
                --num_warmups 5 \
                --num_iterations 20 \
                --compute_type ${ctype}
        done
    done
done

#copy results
#cp *.nvvp /project/projectdirs/mpccc/tkurth/Profiling/tf_perf_kernels/data/good_new/