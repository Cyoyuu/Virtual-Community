import torch

def get_robot_policy(jit_ckpt_path):
	policy = torch.jit.load(jit_ckpt_path)
	policy.to(device='cuda:0')
	return policy