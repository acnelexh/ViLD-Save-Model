docker run -it --gpus all --rm -v /data/datasets/coco:/coco \
-v $PWD:/vild \
daynauth/vild:latest /bin/bash