#!/bin/bash
#SBATCH --job-name=confusion      # Job name
#SBATCH --mail-type=BEGIN,END,FAIL   # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=richardson.t@ufl.edu    # Where to send mail.  Set this to your email address
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept
#SBATCH --ntasks=1                 # Number of MPI tasks (i.e. processes)
#SBATCH --mem=96GB
#SBATCH --time=24:00:00
#SBATCH --array=1-10%50
#SBATCH --output=log/conf_array.%a.out     # Path to the standard output file relative to working directory
#SBATCH --error=log/conf_array.%a.err

python confusion_arrays.py -r $(( $SLURM_ARRAY_TASK_ID - 1 ))
