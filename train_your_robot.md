RoboCLIPv2 lets you use visual reward functions quickly and easily on any robot!

After you have collected your own data in `trajs.h5` and given an encoder provided by `roboclip_model.pth`, 
we can train a roboCLIP reward function transformation:
```
TODO:
python scripts/train_roboclip.py --input trajs.h5 --encoder roboclip_model.pth --output transformation.npy
# Output: -> transformation.npy
```

Then, we will label rewards onto the provided .h5 file
```
python scripts/label_rewards.py --trajs_to_label trajs.h5 --encoder_type xclip --transform_model_path transformation.pth --out updated_trajs.h5
```

Now we can use offline RL to train online
```
TODO:
python scripts/train_online.py --trajs_for_buffer updated_trajs.h5 --encoder roboclip_model.pth --transform transformation.npy
```