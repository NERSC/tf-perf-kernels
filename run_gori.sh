#!/bin/bash
#SBATCH -J conv2d_test
#SBATCH -t 02:00:00
#SBATCH -A nstaff
#SBATCH -C gpu
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=enforce-binding
#SBATCH --exclusive

#load modules
module load cuda
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

#step in
cd ${run_dir}

#list of metrics
#metrics="time"
#metrics="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
#lts__t_sectors_aperture_sysmem_op_read.sum,lts__t_sectors_aperture_sysmem_op_write.sum,\
#dram__sectors_read.sum,dram__sectors_write.sum,\
#lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,\
#l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"

  
${sruncmd} ${profilestring} $(which python) -u ./conv2d.py \
    --dtype float32 \
    --data_format "NHWC" \
    --input_tensor 16 512 512 3 \
    --kernel_shape 5 5 3 32 \
    --compute_type "forward"


