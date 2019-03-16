#!/bin/bash
#SBATCH -J conv2d_test
#SBATCH -t 02:00:00
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --exclusive

#load modules
module load cuda/10.0
module load nccl
module load python3/3.6-anaconda-4.4

#activate env
source activate thorstendl-gori-py3-tf

#rankspernode
rankspernode=1

#openmp stuff
export OMP_NUM_THREADS=$(( 40 / ${rankspernode} ))
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
sruncmd="srun --mpi=pmi2 -N ${SLURM_NNODES} -n $(( ${SLURM_NNODES} * ${rankspernode} )) -c $(( 80 / ${rankspernode} )) --cpu_bind=cores"

#create run dir
run_dir=$WORK/tf_cnn_kernels/runs/${SLURM_JOBID}
mkdir -p ${run_dir}

#copy relevant files
cp conv2d.py ${run_dir}/

#variables
prec=32
batch_size=16
#input_tensor_shape="16-512-512-3"
#kernel_shape="5-5-3-32"
data_format="NHWC"

#net_params
#net_params="VGG-1,224x224x3,3x3x3x64,1 ResNet50-1,224x224x3,7x7x3x64,2"
net_params="VGG-1,224x224x3,3x3x3x64,1"


#step in
cd ${run_dir}

#list of metrics
#metrics="time"
#metrics="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
#lts__t_sectors_aperture_sysmem_op_read.sum,lts__t_sectors_aperture_sysmem_op_write.sum,\
#dram__sectors_read.sum,dram__sectors_write.sum,\
#lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
#l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"
#metrics="time flop_count_sp sysmem_read_transactions sysmem_write_transactions dram_read_transactions dram_write_transactions l2_read_transactions l2_write_transactions"
#metrics="flop_count_sp"
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
	for ctype in forward backward; do

            #get better metric name
            metricname=${metric//,/-}
    
            #assemble profiling string
            if [ "${metric}" == "time" ]; then
		#profilestring="nvprof"
		profilestring="nv-nsight-cu-cli"
            else
		#profilestring="nvprof --replay-mode application --metrics ${metric}"
		profilestring="nv-nsight-cu-cli --metrics ${metric}"
            fi
            #profilestring=${profilestring}" -f -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}.metric_${metricname}.nvvp"
	    profilestring=${profilestring}" -f -o profile.name_${name}.batchsize_${batch_size}.inputshape_${2}.kernelshape_${3}.stride_${4}.dataformat_${data_format}.fp${prec}.pass_${ctype}.metric_${metricname}.nsight"

            #forward
            ${sruncmd} ${profilestring} $(which python) -u ./conv2d.py \
		--dtype float${prec} \
		--data_format ${data_format} \
		--input_tensor_shape ${batch_size} ${input_tensor_shape} \
		--kernel_shape ${kernel_shape} \
		--stride ${stride} \
		--num_warmups 1 \
		--num_iterations 10 \
	        --compute_type ${ctype}
	done
    done
done
