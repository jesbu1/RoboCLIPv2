BASEDIR=$(pwd)

singularity \
        --quiet \
        exec \
        --ipc \
        --nv \
        --cleanenv \
        --pid \
        --bind \
	$BASEDIR:/workdir/RoboCLIPv2,./tmp:/root \
	roboclip.sif \
	/bin/bash
