docker run -it --gpus all --rm -v /data/datasets/:/datasets \
-v $PWD:/vild \
daynauth/vild:latest /bin/bash