#! /bin/bash -x
#SBATCH --account=exalab
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=18:00:00
#SBATCH --partition=dc-mi200
#SBATCH --output=mi100-%j.out
#SBATCH  --error=mi100_error-%j.err
 


conda init
conda activate
python3 --version
export PYTHONPATH=/opt/rocm/libexec/rocm_smi/:$PYTHONPATH
python3 /p/scratch/deepacf/ENS10_ERA5/ens10-d3.7/baselines/get_power_amd.py 

