import torch
from rocket import Rocket
from policy import ActorCritic
import os
import glob

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device: ', device)

if __name__ == '__main__':

    task = 'landing'  # 'hover' or 'landing'
    max_steps = 800
    ckpt_files = sorted(glob.glob(os.path.join(task+'_ckpt', '*.pt')))
    if not ckpt_files:
        print(f'No checkpoint found in {task}_ckpt/. Train first with train_parallel.py')
        exit(1)
    ckpt_dir = ckpt_files[-1]
    print(f'Loading checkpoint: {ckpt_dir}')

    env = Rocket(task=task, max_steps=max_steps)
    net = ActorCritic(input_dim=env.state_dims, output_dim=env.action_dims).to(device)
    if os.path.exists(ckpt_dir):
        checkpoint = torch.load(ckpt_dir, weights_only=False)
        net.load_state_dict(checkpoint['model_G_state_dict'])

    state = env.reset()
    for step_id in range(max_steps):
        action, log_prob, value = net.get_action(state)
        state, reward, done, _ = env.step(action)
        env.render(window_name='test')
        if env.already_crash:
            break

