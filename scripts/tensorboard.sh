bash ./scripts/load_modules.sh
source recipe/bin/activate
tensorboard --logdir=/home/acf15379bv/nerf-reptile/logs --port 6100 --host 0.0.0.0
