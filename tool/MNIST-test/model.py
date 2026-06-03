import torch
import torch.nn as nn
import os
# 定义与训练时相同的模型结构以便加载 state_dict
class NeuralNetwork(nn.Module):
	def __init__(self):
		super().__init__()
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU(),
			nn.Linear(512, 512),
			nn.ReLU(),
			nn.Linear(512, 10),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits


MODEL_PATH = { "DNN": os.path.join("data", "models", "model-DNN.pth") }



def load_model(device, model_TYPE):
	model = NeuralNetwork().to(device)
	if os.path.exists(MODEL_PATH.get(model_TYPE)):
		state = torch.load(MODEL_PATH.get(model_TYPE), map_location=device)
		try:
			model.load_state_dict(state)
			print("Loaded model state_dict from", MODEL_PATH.get(model_TYPE))
		except Exception as e:
			print("Failed to load state_dict:", e)
	else:
		print("Model file not found at:", MODEL_PATH.get(model_TYPE))
	model.eval()
	return model
