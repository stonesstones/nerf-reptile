#!/bin/bash
SRN_CAR="https://drive.google.com/file/d/19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU/view?usp=drive_link"
DATADIR="/home/acf15379bv/nerf-reptile/data"
TEMPFILE="/home/acf15379bv/nerf-reptile/data/tmp.zip"
https://drive.google.com/file/d/19yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU/view?usp=sharing
mkdir -p DATADIR
mv

FILE_ID=9yDsEJjx9zNpOKz9o6AaK-E8ED6taJWU
FILE_NAME=TEMPFILE
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}

unzip $TEMPFILE -d $DATADIR && rm -f $TEMPFILE
https://drive.google.com/file/d/0B4y35FiV1wh7SDd1Q1dUQkZQaUU/view?usp=sharing&resourcekey=0-jaSgi7h7yBzLh00gI2TdxA
