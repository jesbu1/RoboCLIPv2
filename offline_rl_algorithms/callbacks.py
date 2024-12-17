import wandb

from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import imageio
import wandb
from wandb.integration.sb3 import WandbCallback
import io

class OfflineEvalCallback(EvalCallback):
    def __init__(self, *args, video_freq, **kwargs):
        super(OfflineEvalCallback, self).__init__(*args, **kwargs)
        self.video_freq = video_freq
        # we need to overide num_timesteps as EvalCallback uses it to align the built in logger's x-axis
        # we are using wandb so now we're using self.n_calls as the step for everything
        self.num_timesteps = lambda x: self.n_calls  # convert num_timst

    def _on_step(self) -> bool:
        # print(self.n_calls, self.n_calls % self.video_freq)
        # Log policy gradients
        policy_gradients = [
            param.grad.view(-1).detach().cpu().numpy()  # Flatten each gradient tensor
            for param in self.model.policy.actor.parameters()
            if param.grad is not None
        ]
        if len(policy_gradients) != 0:

            all_gradients = np.concatenate(policy_gradients)
            self.logger.record("grad/policy_histogram", wandb.Histogram(all_gradients))
            # Log critic gradients
            critic_gradients = [
                param.grad.view(-1)
                .detach()
                .cpu()
                .numpy()  # Flatten each gradient tensor
                for param in self.model.policy.critic.parameters()
                if param.grad is not None
            ]
            if len(critic_gradients) != 0:
                all_gradients = np.concatenate(critic_gradients)
                self.logger.record(
                    "grad/critic_histogram", wandb.Histogram(all_gradients)
                )

            # Log critic_target gradients
            critic_target_gradients = [
                param.grad.view(-1)
                .detach()
                .cpu()
                .numpy()  # Flatten each gradient tensor
                for param in self.model.policy.critic_target.parameters()
                if param.grad is not None
            ]
            if len(critic_target_gradients) != 0:
                all_gradients = np.concatenate(critic_target_gradients)
                self.logger.record(
                    "grad/critic_target_histogram", wandb.Histogram(all_gradients)
                )

            # Log critic weights
            critic_weights = [
                param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
                for param in self.model.policy.critic.parameters()
            ]
            if len(critic_weights) != 0:
                all_weights = np.concatenate(critic_weights)
                self.logger.record(
                    "weights/critic_histogram", wandb.Histogram(all_weights)
                )

        # Log critic_target weights
        critic_target_weights = [
            param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
            for param in self.model.policy.critic_target.parameters()
        ]
        if len(critic_target_weights) != 0:
            all_weights = np.concatenate(critic_target_weights)
            self.logger.record(
                "weights/critic_target_histogram", wandb.Histogram(all_weights)
            )
            # Log critic_target weights
            critic_target_weights = [
                param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
                for param in self.model.policy.critic_target.parameters()
            ]
            if len(critic_target_weights) != 0:
                all_weights = np.concatenate(critic_target_weights)
                self.logger.record(
                    "weights/critic_target_histogram", wandb.Histogram(all_weights)
                )

        # Log v_net weights and gradients
        if hasattr(self.model, "v_net"):
            # Log v_net gradients
            v_net_gradients = [
                param.grad.view(-1)
                .detach()
                .cpu()
                .numpy()  # Flatten each gradient tensor
                for param in self.model.v_net.parameters()
                if param.grad is not None
            ]
            if len(v_net_gradients) != 0:
                all_gradients = np.concatenate(v_net_gradients)
                self.logger.record(
                    "grad/v_net_histogram", wandb.Histogram(all_gradients)
                )

            # Log v_net weights
            v_net_weights = [
                param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
                for param in self.model.v_net.parameters()
            ]
            if len(v_net_weights) != 0:
                all_weights = np.concatenate(v_net_weights)
                self.logger.record(
                    "weights/v_net_histogram", wandb.Histogram(all_weights)
                )

        # Log policy weights
        actor_weights = [
            param.data.view(-1).detach().cpu().numpy()  # Flatten each weight tensor
            for param in self.model.policy.actor.parameters()
        ]
        if len(actor_weights) != 0:
            all_weights = np.concatenate(actor_weights)
            self.logger.record("weights/policy_histogram", wandb.Histogram(all_weights))

        # breakpoint()
        if (
            self.video_freq > 0 and self.n_calls % self.video_freq == 0
        ) or self.n_calls == 1:
            video_buffer = self.record_video()
            # self.logger.record({f"evaluation_video": wandb.Video(video_buffer, fps=20, format="mp4")}, commit=False)
            self.logger.record(
                "eval/evaluation_video", wandb.Video(video_buffer, fps=20, format="mp4")
            )
            # self.logger.record({f"eval/evaluate_succ": success}, step = self.n_calls)
            print("video logged")

        self.logger.record("num_timesteps", self.num_timesteps)

        result = super(OfflineEvalCallback, self)._on_step()

        return result

    def record_video(self):
        frames = []
        obs = self.eval_env.reset()
        # success = 0
        # breakpoint()

        # print the first layer's weight of self.model.policy
        print(self.model.policy.actor.latent_pi[0].weight[0][:10])

        for _ in range(128):  # You can adjust the number of steps for recording
            frame = self.eval_env.render(mode="rgb_array")
            # downsample frame
            frame = frame[::3, ::3, :3]
            frames.append(frame)
            action, _ = self.model.predict(obs, deterministic=False)
            # action += np.random.normal(5, 0.1, size=action.shape)
            # action[0] += 5
            # print(action)
            obs, _, _, info = self.eval_env.step(action)
            # print(type(info))
            # print(info)
            # if info['success']:
            #     success = 1
            #     break

        video_buffer = io.BytesIO()

        with imageio.get_writer(video_buffer, format="mp4", fps=20) as writer:
            for frame in frames:
                writer.append_data(frame)

        video_buffer.seek(0)
        return video_buffer


class CustomWandbCallback(WandbCallback):
    def _on_step(self):
        if "metrics" in self.locals:
            self.logger.record_dict(self.locals["metrics"])
        self.logger.dump(
            self.n_calls
        )  # this ensures that dump gets called, otherwise it's only called in EvalCallback whenever an eval happens
