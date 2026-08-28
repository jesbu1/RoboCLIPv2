import numpy as np


def build_one_step_next_observations(observations, dones):
    """Infer s_{t+1} from episode-ordered transition rows.

    Terminal rows keep their current observation because the bootstrapped value
    is masked by ``done``. A final non-terminal row is rejected because its next
    observation cannot be inferred from the flattened dataset.
    """
    observations = np.asarray(observations)
    dones = np.asarray(dones).reshape(-1)

    if observations.ndim == 0:
        raise ValueError("observations must have a leading transition dimension")
    if observations.shape[0] != dones.shape[0]:
        raise ValueError(
            "observations and dones must contain the same number of transitions"
        )

    next_observations = observations.copy()
    if observations.shape[0] == 0:
        return next_observations
    if dones[-1] < 0.5:
        raise ValueError(
            "cannot infer next observation for a final non-terminal transition"
        )

    nonterminal_indices = np.flatnonzero(dones[:-1] < 0.5)
    next_observations[nonterminal_indices] = observations[nonterminal_indices + 1]
    return next_observations
