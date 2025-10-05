from stable_baselines3.common.callbacks import BaseCallback

# Custom callback that overwrites one checkpoint
class SingleCheckpointCallback(BaseCallback):
    def __init__(self, save_path, save_freq=500_000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose:
                print(f"✅ Checkpoint saved to {self.save_path} at step {self.num_timesteps}")
        return True