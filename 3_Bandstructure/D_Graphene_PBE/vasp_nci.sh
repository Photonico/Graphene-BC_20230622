#!/bin/bash
#PBS -N job_name
#PBS -P g46
#PBS -q normal
#PBS -o output.txt
#PBS -j oe
#PBS -l mem=190GB
#PBS -l ncpus=48
#PBS -l walltime=47:59:59
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -l software=vasp
#PBS -m ea
#PBS -M luke.niu@sydney.edu.au

module load vasp/6.3.2

mpirun vasp_std >vasp.log