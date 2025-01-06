This file will contain a list of the commands needed to run every kind of experiment. This is also meant to serve as a list of experiments that should run for each method

The structure of a train command is as follows:
```
python test_scripts/test_iql.py metaworld=X algorithm=Y reward=Z
```

Note that the order of `metaworld`, `algorithm`, and `reward` must be in the same as above, as the configs will override each other in that order. Below, I will describe the current options in each one, and they can be found in `configs/`

-  `metaworld=`: Defines metaworld environment configs. All environments are tested on opening a window.
    -  `all_tasks_15`: Training on 50 metaworld tasks with 15 demos each
    -  `all_tasks_100`: All 50 metaworld tasks with 100 demos each
    -  `single_task`: Training and testing on opening a window
    -  `some_tasks_15`: Only 8(?) tasks including opening a window for training
    -  `task_gen`: Train on X tasks while leaving out the eval setting. TODO incompleted (but only requires simple yaml changes)

- `algorithm`: Defines different RL algorithms, mostly on offline algorithms only. These settings should be general enough for all environments (hopefully!). All WSRL and RLPD examples uses the `rlpd.py` code
    - `bc`: behavior cloning offline only
    - `sac`: SAC training online only
    - `iql`: IQL training offline and online, though you should not expect good performance online
    - `cql`: CQL training offline and online, and you should expect a somewhat large dropoff when going online
    - `rlpd`: RLPD does not train offline and instead instantiates an offline buffer to use during online training
    - `rlpd_scratch`: Train the RLPD class online without an offline buffer. This is meant to test the training code without any pretraining or offline
    - `rlpd_iql`: Train the policy and critics offline with IQL offline, then use the offline data and keep training the policy online
    - `rlpd_calql`: Train the policy and critics offline with Cal-QL offline, then use the offline data and keep training the policy online
    - `wsrl_iql`: Train the policy and critics offline using IQL, then throw away the offline data, rollout the current policy N times on the new task to warmstart finetuning online
    - `wsrl_calql`: Train the policy and critics offline using IQL, then throw away the offline data, rollout the current policy N times on the new task to warmstart finetuning online


- `reward`: Defines different reward strategies and uses its associated visual encoder when using image information as opposed to state
    - `dense`: The default environment return returns nothing and uses the dense envirnoment reward. Only useful in simulation.
    - `sparse`: The default environment returns nothing and uses the success provided by success 
    - `roboclipv2`: RoboCLIPv2 reward model and a LIV image/text encoder for observation encoding
    - `roboclip`: A zero-shot RoboCLIP reward model and an S3D image/text encoder for observatiopn
    - `vlc`: VLC reward model and a LIV image/text encoder for observation encoding.

If you want to use state-based observations, you can use add the following arg to render less often to speed up training:
`environment.is_state_based=true`. However, by default, ALL configs assume that you are using images, so use that argument to ensure you are using offline.


To run RLPD on roboclipv2 reward on some tasks, run:
```
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=rlpd reward=roboclipv2
```

Then to run RLPD on roboclipv2 reward using a state-based observation, you simply need to run:
```
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=rlpd reward=roboclipv2  environment.is_state_based=true
```

To run IQL using sparse reward, run:
```
python test_scripts/test_iql.py metaworld=some_tasks_15 algorithm=iql reward=sparse  environment.is_state_based=true
```