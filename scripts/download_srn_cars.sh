#!/bin/bash
SRN_CAR="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_3.zip"
DATADIR="/home/acf15379bv/nerf-reptile/data"
TEMPFILE="/home/acf15379bv/nerf-reptile/data/tmp.zip"

mkdir -p DATADIR
wget $RIGHT -O $TEMPFILE && unzip $TEMPFILE -d $DATADIR && rm -f $TEMPFILE
