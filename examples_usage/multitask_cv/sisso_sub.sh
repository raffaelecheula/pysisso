#!/bin/bash

#SBATCH --job-name=pysisso
#SBATCH --partition=q48
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6G
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --error=job.err
#SBATCH --output=job.out

module load intel/2020.4
module load openmpi/4.0.3
ulimit -s unlimited

scratch_run=false
out_file=pysisso_out.txt

calc_dir=$SLURM_SUBMIT_DIR
if [ $scratch_run = true ]; then
    cp -r * /scratch/$SLURM_JOB_ID
    cd /scratch/$SLURM_JOB_ID
    [ -e $out_file ] && rm $out_file
fi

echo "========= Job started  at `date` ==========" >> $calc_dir/job.out

python sisso_regression.py > $calc_dir/$out_file

echo "========= Job finished at `date` ==========" >> $calc_dir/job.out

if [ $scratch_run = true ]; then
    cp -r * $calc_dir
    cd $calc_dir
fi