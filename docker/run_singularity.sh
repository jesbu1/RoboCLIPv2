BASEDIR=$(pwd)

singularity \
        --quiet \
        exec \
        --ipc \
        --nv \
        --cleanenv \
        --pid \
        --bind \
	$BASEDIR:/workdir/RoboCLIPv2,./tmp:$HOME \
	roboclip.sif \
	/bin/bash
