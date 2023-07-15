#!/bin/bash

#$-l rt_G.small=1
#$-l h_rt=8:00:00
#$-j y
#$ -o ./log4txt
#$-cwd

source /etc/profile.d/modules.sh
module load singularitypro
SINGULARITY_TMPDIR=$SGE_LOCALDIR singularity exec --nv \
--bind /home/acf15379bv/nerf_reptile:/home/acf15379bv/nerf_reptile \
/home/acf15379bv/nerf_reptile/docker/nerfstudio2.sif \
bash scripts/train_nerf.sh
