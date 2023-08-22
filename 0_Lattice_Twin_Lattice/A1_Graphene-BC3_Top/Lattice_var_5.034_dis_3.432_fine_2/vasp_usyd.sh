#!/bin/csh
#PBS -N lattice
#PBS -q cmt
#PBS -j oe
#PBS -l select=1:ncpus=16:mpiprocs=16:mem=64GB
#PBS -l walltime=48:00:00
#PBS -m a
#PBS -M luke.niu@sydney.edu.au

cd "$PBS_O_WORKDIR"

module purge
module load pbspro-intelmpi 
module load compiler-rt mpi mkl
set VASP=~/VASP544/bin/vasp_std
set BIN=~/VASP544/bin/vasp_std

mpirun -np 16 $VASP > vasp.out
