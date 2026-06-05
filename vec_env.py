import numpy as np
from rocket import Rocket


class VecEnv:
    """
    Synchronous vectorized environment: runs N Rocket instances in lockstep.
    Auto-resets environments that reach a terminal state.
    Only env[0] is set up for rendering.
    """

    def __init__(self, num_envs, max_steps, task='hover', rocket_type='falcon'):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.task = task

        self.envs = []
        for i in range(num_envs):
            env = Rocket(
                max_steps=max_steps,
                task=task,
                rocket_type=rocket_type,
                viewport_h=768 if i == 0 else 1,
            )
            self.envs.append(env)

        # Patch non-rendered envs to avoid cv2.destroyAllWindows() on reset
        for i in range(1, num_envs):
            self._patch_reset(self.envs[i])

        self.state_dims = self.envs[0].state_dims
        self.action_dims = self.envs[0].action_dims

    @staticmethod
    def _patch_reset(env):
        def patched_reset(state_dict=None):
            if state_dict is None:
                env.state = env.create_random_state()
            else:
                env.state = state_dict
            env.state_buffer = []
            env.step_id = 0
            env.already_landing = False
            env.already_crash = False
            return env.flatten(env.state)
        env.reset = patched_reset

    def reset(self):
        states = np.zeros((self.num_envs, self.state_dims), dtype=np.float32)
        for i, env in enumerate(self.envs):
            states[i] = env.reset()
        return states

    def step(self, actions):
        states = np.zeros((self.num_envs, self.state_dims), dtype=np.float32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos = [{} for _ in range(self.num_envs)]

        for i, env in enumerate(self.envs):
            state, reward, done, _ = env.step(int(actions[i]))

            if done:
                infos[i]['episode_reward'] = reward
                infos[i]['episode_done'] = True
                state = env.reset()

            states[i] = state
            rewards[i] = reward
            dones[i] = done

        return states, rewards, dones, infos

    def render(self, env_id=0, **kwargs):
        return self.envs[env_id].render(**kwargs)
