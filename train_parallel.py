import numpy as np
import torch
from vec_env import VecEnv
from policy import ActorCritic
import utils
import os
import glob
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

if __name__ == '__main__':

    task = 'landing'            # 'hover' or 'landing'
    num_envs = 16               # parallel environments
    max_steps = 800             # max steps per episode
    rollout_steps = 256         # steps collected before each update
    max_updates = 50000         # total gradient updates
    gamma = 0.999
    gae_lambda = 0.95
    ppo_epochs = 4
    mini_batch_size = 512
    clip_eps = 0.2
    ent_coef = 0.01
    max_grad_norm = 0.5
    render_interval = 100       # render env[0] every N updates
    save_interval = 100         # save checkpoint every N updates

    wandb.init(
        project="rocket-recycling",
        config={
            "task": task,
            "algorithm": "PPO",
            "num_envs": num_envs,
            "max_steps": max_steps,
            "rollout_steps": rollout_steps,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "ppo_epochs": ppo_epochs,
            "mini_batch_size": mini_batch_size,
            "clip_eps": clip_eps,
            "ent_coef": ent_coef,
            "max_grad_norm": max_grad_norm,
        },
    )

    vec_env = VecEnv(num_envs=num_envs, max_steps=max_steps, task=task)
    ckpt_folder = os.path.join('./', task + '_ckpt')
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    net = ActorCritic(input_dim=vec_env.state_dims, output_dim=vec_env.action_dims).to(device)

    last_update_id = 0
    REWARDS = []
    best_mean_reward = -float('inf')
    episode_rewards = np.zeros(num_envs)

    ckpt_files = sorted(glob.glob(os.path.join(ckpt_folder, '*.pt')))
    if len(ckpt_files) > 0:
        checkpoint = torch.load(ckpt_files[-1], weights_only=False)
        net.load_state_dict(checkpoint['model_G_state_dict'])
        last_update_id = checkpoint.get('update_id', 0)
        REWARDS = checkpoint.get('REWARDS', [])
        best_mean_reward = checkpoint.get('best_mean_reward', -float('inf'))

    states = vec_env.reset()

    for update_id in range(last_update_id, max_updates):

        mb_states = np.zeros((rollout_steps, num_envs, vec_env.state_dims), dtype=np.float32)
        mb_actions = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        mb_rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        mb_masks = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        mb_log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        mb_values = np.zeros((rollout_steps, num_envs), dtype=np.float32)

        should_render = (update_id % render_interval == 0)

        for t in range(rollout_steps):
            with torch.no_grad():
                actions, log_probs, values = net.get_actions_batch(states)

            next_states, rewards, dones, infos = vec_env.step(actions)

            mb_states[t] = states
            mb_actions[t] = actions
            mb_rewards[t] = rewards
            mb_masks[t] = 1.0 - dones.astype(np.float32)
            mb_log_probs[t] = log_probs.cpu().numpy()
            mb_values[t] = values.cpu().numpy()

            episode_rewards += rewards
            for i in range(num_envs):
                if dones[i]:
                    REWARDS.append(episode_rewards[i])
                    wandb.log({
                        "reward/episode": episode_rewards[i],
                        "episode": len(REWARDS),
                    })
                    episode_rewards[i] = 0.0

            if should_render:
                vec_env.render(env_id=0)

            states = next_states

        # Bootstrap value for final states
        with torch.no_grad():
            _, _, bootstrap_values = net.get_actions_batch(states)

        # Compute GAE
        mb_rewards_t = torch.tensor(mb_rewards, dtype=torch.float32).to(device)
        mb_values_t = torch.tensor(mb_values, dtype=torch.float32).to(device)
        mb_masks_t = torch.tensor(mb_masks, dtype=torch.float32).to(device)

        advantages, returns = ActorCritic.compute_gae(
            mb_rewards_t, mb_values_t, mb_masks_t, bootstrap_values,
            gamma=gamma, lam=gae_lambda
        )

        # Flatten [T, N, ...] -> [T*N, ...]
        flat_states = torch.tensor(
            mb_states.reshape(-1, vec_env.state_dims), dtype=torch.float32
        ).to(device)
        flat_actions = torch.tensor(mb_actions.reshape(-1), dtype=torch.int64).to(device)
        flat_old_log_probs = torch.tensor(mb_log_probs.reshape(-1), dtype=torch.float32).to(device)
        flat_returns = returns.view(-1)
        flat_advantages = advantages.view(-1)

        # PPO update
        ActorCritic.update_ppo(
            net, flat_states, flat_actions, flat_old_log_probs,
            flat_returns, flat_advantages,
            ppo_epochs=ppo_epochs, mini_batch_size=mini_batch_size,
            clip_eps=clip_eps, ent_coef=ent_coef, max_grad_norm=max_grad_norm
        )

        # Logging
        rollout_mean_reward = mb_rewards.sum(axis=0).mean()
        wandb.log({
            "rollout/mean_reward": rollout_mean_reward,
            "rollout/mean_value": mb_values.mean(),
            "rollout/mean_advantage": flat_advantages.mean().item(),
            "update": update_id,
        })

        if len(REWARDS) > 0 and update_id % 10 == 0:
            recent = REWARDS[-num_envs * 10:] if len(REWARDS) > num_envs * 10 else REWARDS
            mean_reward = np.mean(recent)
            print(f'update {update_id}, episodes: {len(REWARDS)}, '
                  f'mean reward (recent): {mean_reward:.3f}')

        # Save checkpoint
        if update_id % save_interval == 0 and update_id > 0:
            torch.save({
                'update_id': update_id,
                'episode_id': len(REWARDS),
                'REWARDS': REWARDS,
                'best_mean_reward': best_mean_reward,
                'model_G_state_dict': net.state_dict()
            }, os.path.join(ckpt_folder, f'ckpt_{update_id:08d}.pt'))

        # Save best checkpoint
        if len(REWARDS) >= 50:
            recent_mean = np.mean(REWARDS[-50:])
            if recent_mean > best_mean_reward:
                best_mean_reward = recent_mean
                torch.save({
                    'update_id': update_id,
                    'episode_id': len(REWARDS),
                    'REWARDS': REWARDS,
                    'best_mean_reward': best_mean_reward,
                    'model_G_state_dict': net.state_dict()
                }, os.path.join(ckpt_folder, 'best.pt'))
                print(f'  -> new best model saved (mean reward: {best_mean_reward:.3f})')

    wandb.finish()
