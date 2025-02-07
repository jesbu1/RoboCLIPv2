import h5py



train_keys = [
    'Move the orange cup from the left to the right', 
    'Move the orange cup from the right to the left',

    'Put the orange cup on the red plate',
    'Put the red cup on the red plate',

    'Separate the blue and red cups',

    'Fold the blue towel',

    'Open the green trash bin',
    'Open the blue trash bin',

    'Throw the banana away in the green trash bin',
    'Throw the banana away in the blue trash bin',

    'Put the red marker in the red trash can',
    'Put the pink marker in the green trash can',

    'Put the blue tape in the box on the left',

    'Put the banana in the box',
    'Put the orange cup in the box'
]

Eval_keys = [
    'Put the blue cup on the red plate',
    'Separate the orange and blue cups',
    'Open the red trash bin',
    'Throw the banana away in the red trash bin',
    'Put the red tape in the box on the right'
]

h5_file_name = "usc_koch_rewind_reward.h5"
h5_file = h5py.File(h5_file_name, 'r')

# print("Keys: %s" % h5_file.keys())

for key in train_keys:
    if key not in h5_file:
        print("Key not found: ", key)

for key in Eval_keys:
    if key not in h5_file:
        print("Key not found: ", key)


train_file = h5py.File("usc_koch_rewind_reward_train.h5", 'w')

for key in train_keys:
    h5_file.copy(key, train_file)

train_file.close()
eval_file = h5py.File("usc_koch_rewind_reward_eval.h5", 'w')
for key in Eval_keys:
    h5_file.copy(key, eval_file)
