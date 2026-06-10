import torch
import torch.nn as nn
import os
# 定义与训练时相同的模型结构以便加载 state_dict
class NeuralNetwork_DNN(nn.Module):
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
	
class NeuralNetwork_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv= nn.Sequential( #MNIST :28*28*1 ->10
            nn.Conv2d(1,28,3,1,padding=1), #=>28*28*28
            nn.Sigmoid(),
            nn.MaxPool2d(2,2), #14x14x28
            nn.Conv2d(28,8,3,1,padding=1), #=>14*14*8
            nn.Sigmoid(),
            nn.MaxPool2d(2,2), #7*7*8
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential( #MNIST :28*28*1 ->10
            nn.Linear(7*7*8,88),
            nn.Sigmoid(),
            nn.Linear(88,10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


MODEL_PATH = {"DNN": os.path.join("data", "models", "model-DNN.pth"),
			  "CNN": os.path.join("data", "models", "model-CNN.pth") }

MODELTYPE = {"DNN": NeuralNetwork_DNN, "CNN": NeuralNetwork_CNN}


def load_model(device, model_TYPE):
	model = MODELTYPE.get(model_TYPE)().to(device)
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
