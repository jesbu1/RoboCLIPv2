# BUILDING
To re-build, uncomment `#- mujoco-py==2.1.2.14` in`roboclipv2.yml` and comment out the other mujoco thing.
# RUNNING
simply run:

git clone https://github.com/sumedh7/RoboCLIP.git --recursive
cd RoboCLIP
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_howto100m.pth
wget https://www.rocq.inria.fr/cluster-willow/amiech/howto100m/s3d_dict.npy
singularity pull roboclipv2.sif docker://jesbu1/roboclipv2
singularity exec roboclipv2.sif .........