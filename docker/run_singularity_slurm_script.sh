BASEDIR=$(pwd)
ROBOCLIP_LOC=/workdir/RoboCLIPv2
SCRIPT=$1

singularity \
        --quiet \
        exec \
        --ipc \
        --nv \
        --cleanenv \
        --pid \
        --bind \
	$BASEDIR:$ROBOCLIP_LOC,./tmp:/root, \
	roboclip.sif \
        /bin/bash -c "cd $ROBOCLIP_LOC && conda init bash && source ~/.bashrc && conda activate roboclip && $SCRIPT"