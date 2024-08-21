BASEDIR=$(pwd)

singularity \
        --quiet \
        exec \
        --ipc \
        --nv \
        --cleanenv \
        --pid \
        --bind \
	$BASEDIR:/workdir/RoboCLIPv2 \
	docker:jesbu1/roboclipv2 \
	/bin/bash