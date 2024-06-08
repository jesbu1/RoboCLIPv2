from gym.envs.registration import register

register(
    id='kitchen-sparse-v0',
    entry_point='kitchen.envs:KitchenEnvSparseOriginalReward',
)
register(
    id='kitchen-dense-v0',
    entry_point='kitchen.envs:KitchenEnvDenseOriginalReward',
)
register(
    id='kitchen-sparse-img-v0',
    entry_point='kitchen.envs:KitchenEnvSparseOriginalRewardImage',
)
register(
    id='kitchen-sparse-learnt-v0',
    entry_point='kitchen.envs:KitchenEnvSparseReward',
)
