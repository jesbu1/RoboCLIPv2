docker run -it --network=host --gpus=all \
    -v ./:/workdir/RoboCLIPv2 \
    jesbu1/roboclipv2 /bin/bash