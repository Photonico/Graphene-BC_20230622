#!/bin/bash 
#PBS -N dielectric_function
#PBS -P MOSHEMT
#PBS -q defaultQ
#PBS -o output.txt
#PBS -j oe
#PBS -l select=1:ncpus=24:mem=100GB
#PBS -l walltime=24:00:00
#PBS -m ea
#PBS -M lniu6305@uni.sydney.edu.au

cd "$PBS_O_WORKDIR"

module purge
module load intel-mpi/18.1 hdf5-intel/1.10.6

mpirun -np 24 /project/RDS-FSC-MOSHEMT-RW/vasp6/vasp_std
E=`tail -1 OSZICAR`
echo $E >> ENERGY.dat
