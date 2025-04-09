#!/bin/sh
#SBATCH -n 16     # 16 cores
#SBATCH -t 1-03:00:00   # 1 day and 3 hours
#SBATCH -p compute      # partition name
#SBATCH -J poisson2d_15  # sensible name for the job
#SBATCH --output=poisson_2d_15_nprocs16_%j.out  # Output file name, %j will be replaced by the job ID


# [load up the correct modules, if required](#load-up-the-correct-modules-if-required)
module load openmpi-3.1.6-gcc-9.3.0

cd /home/users/mschpc/2024/lipsiuci/


# [launch the code](#launch-the-code)
# /home/support/apps/intel/rhel7/19.0.5/compilers_and_libraries_2019.5.281/linux/mpi/intel64/bin/mpirun -n 16 ./poiss2d 15
mpirun -n 16 ./poiss2d 31
