"""MNIST 手写数字绘图和检测 (Pygame)

功能：
- 用 28x28 网格绘制数字（可缩放显示）
- 笔、橡皮、清空操作
- 生成灰度图并进行模型推理（加载 ./data/models/model.pth）
- 显示 0-9 的概率条形图

用法：
python MNIST_cheek.py
"""

import os
import sys
import pygame
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

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


MODEL_PATH = os.path.join("..", "data", "models", "model.pth")
LOCAL_MODEL_PATH = os.path.join("data", "models", "model.pth")
if os.path.exists(LOCAL_MODEL_PATH):
	MODEL_PATH = LOCAL_MODEL_PATH


def load_model(device):
	model = NeuralNetwork().to(device)
	if os.path.exists(MODEL_PATH):
		state = torch.load(MODEL_PATH, map_location=device)
		try:
			model.load_state_dict(state)
			print("Loaded model state_dict from", MODEL_PATH)
		except Exception as e:
			print("Failed to load state_dict:", e)
	else:
		print("Model file not found at:", MODEL_PATH)
	model.eval()
	return model


def nparr_to_tensor(img28):
	# img28: 28x28 uint8, 0..255, background 0, stroke 255
	arr = np.array(img28, dtype=np.float32) / 255.0
	# ensure shape (1,1,28,28)
	t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
	return t


def run_pygame():
	pygame.init()
	SCALE = 20  # 每个 28x28 网格的像素放大倍数
	GRID = 28
	DRAW_SIZE = GRID * SCALE
	INFO_WIDTH = 360
	WIN_W = DRAW_SIZE + INFO_WIDTH
	WIN_H = max(DRAW_SIZE, 360)

	screen = pygame.display.set_mode((WIN_W, WIN_H))
	pygame.display.set_caption("MNIST 手写绘制与检测")

	font = pygame.font.Font("pytorch文档\MNIST-test\msyh.ttc", 20)
	bigfont = pygame.font.Font("pytorch文档\MNIST-test\msyh.ttc", 28)

	# 网格数据：0 黑背景，255 白笔画
	grid = np.zeros((GRID, GRID), dtype=np.uint8)

	drawing = False
	tool = 'pen'  # 'pen' or 'eraser'
	pen_value = 255
	eraser_value = 0
	# 灰度画笔强度 0..255，可用鼠标滚轮或上下键调整
	brush_intensity = 255

	# 画笔半径（以网格单元为单位），用于影响周围格子的灰度
	brush_radius = 2

	def apply_brush(grid, cx, cy, intensity, radius, tool):
		h, w = grid.shape
		r = int(max(0, radius))
		# 使用平方衰减: factor = (max(0, 1 - d/r))**2
		for dy in range(-r, r+1):
			for dx in range(-r, r+1):
				x = cx + dx
				y = cy + dy
				if x < 0 or x >= w or y < 0 or y >= h:
					continue
				dist = (dx*dx + dy*dy) ** 0.5
				if dist > r:
					continue
				if r == 0:
					factor = 1.0
				else:
					factor = max(0.0, 1.0 - (dist / r)) ** 2
				val = int(intensity * factor)
				if tool == 'pen':
					# 叠加取最大值，使多次涂抹更深
					grid[y, x] = max(int(grid[y, x]), val)
				else:
					# 橡皮擦按比例擦除（线性）
					grid[y, x] = int(grid[y, x] * (1.0 - factor))

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = load_model(device)

	probs = None

	clock = pygame.time.Clock()
	running = True
	while running:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				mx, my = event.pos
				if mx < DRAW_SIZE and my < DRAW_SIZE:
					drawing = True
					gx, gy = mx // SCALE, my // SCALE
					if 0 <= gx < GRID and 0 <= gy < GRID:
						apply_brush(grid, gx, gy, brush_intensity, brush_radius, tool)
				else:
					# 点击右侧按钮区域
					bx = mx - DRAW_SIZE
					by = my
					# 按钮区域布局
					if 20 <= bx <= 120 and 20 <= by <= 60:
						tool = 'pen'
					elif 140 <= bx <= 240 and 20 <= by <= 60:
						tool = 'eraser'
					elif 260 <= bx <= 340 and 20 <= by <= 60:
						grid[:, :] = 0
						probs = None
					elif 20 <= bx <= 160 and 80 <= by <= 120:
						# Detect
						img = Image.fromarray(grid).resize((28, 28), Image.BILINEAR)
						t = nparr_to_tensor(np.array(img))
						t = t.to(device)
						with torch.no_grad():
							out = model(t)
							soft = torch.softmax(out, dim=1).cpu().numpy().flatten()
							probs = soft.tolist()
                    
			elif event.type == pygame.MOUSEBUTTONUP:
				drawing = False
			elif event.type == pygame.MOUSEWHEEL:
				# 鼠标滚轮调整画笔强度（每格 16）
				brush_intensity = int(np.clip(brush_intensity + event.y * 16, 0, 255))
			elif event.type == pygame.KEYDOWN:
				# 上下键调整强度
				if event.key == pygame.K_UP:
					brush_intensity = int(np.clip(brush_intensity + 16, 0, 255))
				elif event.key == pygame.K_DOWN:
					brush_intensity = int(np.clip(brush_intensity - 16, 0, 255))
				elif event.key == pygame.K_RIGHTBRACKET:
					brush_radius = int(np.clip(brush_radius + 1, 0, GRID))
				elif event.key == pygame.K_LEFTBRACKET:
					brush_radius = int(np.clip(brush_radius - 1, 0, GRID))
			elif event.type == pygame.MOUSEMOTION and drawing:
				mx, my = event.pos
				if mx < DRAW_SIZE and my < DRAW_SIZE:
					gx, gy = mx // SCALE, my // SCALE
					if 0 <= gx < GRID and 0 <= gy < GRID:
						apply_brush(grid, gx, gy, brush_intensity, brush_radius, tool)

		# 绘制画布：先在 28x28 小 surface 上写入像素，再缩放到显示尺寸
		screen.fill((30, 30, 30))
		small_surf = pygame.Surface((GRID, GRID))
		# grid 的值 0..255，对应黑到白。转换成 RGB
		rgb = np.stack([grid]*3, axis=2)
		# blit_array 需要数组与 surface 大小一致（width,height,3）
		pygame.surfarray.blit_array(small_surf, rgb.transpose((1, 0, 2)))
		scaled = pygame.transform.scale(small_surf, (DRAW_SIZE, DRAW_SIZE))
		screen.blit(scaled, (0, 0))

		# 网格线
		for i in range(GRID+1):
			x = i * SCALE
			pygame.draw.line(screen, (50, 50, 50), (x, 0), (x, DRAW_SIZE))
			y = i * SCALE
			pygame.draw.line(screen, (50, 50, 50), (0, y), (DRAW_SIZE, y))

		# 右侧 UI
		ui_x = DRAW_SIZE
		pygame.draw.rect(screen, (40, 40, 40), (ui_x, 0, INFO_WIDTH, WIN_H))

		# 按钮
		def draw_button(x, y, w, h, label, active=False):
			color = (80, 160, 80) if active else (120, 120, 120)
			pygame.draw.rect(screen, color, (ui_x + x, y, w, h))
			txt = font.render(label, True, (0, 0, 0))
			screen.blit(txt, (ui_x + x + 8, y + 8))

		draw_button(20, 20, 100, 40, '画笔', tool == 'pen')
		draw_button(140, 20, 100, 40, '橡皮', tool == 'eraser')
		draw_button(260, 20, 80, 40, '清空')
		draw_button(20, 80, 140, 40, '检测')

		# 显示缩略图和数值
		thumb = pygame.transform.scale(scaled, (140, 140))
		screen.blit(thumb, (ui_x + 200, 20))

		# 若有概率结果，绘制条形图（增加每项间距以防挤在一起）
		if probs is not None:
			# 绘制标题
			title = bigfont.render('预测概率', True, (220, 220, 220))
			screen.blit(title, (ui_x + 20, 140))
			# 绘制每个数字的条
			max_w = INFO_WIDTH - 120
			prob_start_y = 180
			prob_spacing = 26  # 间距增大
			for i, p in enumerate(probs):
				by = prob_start_y + i * prob_spacing
				label = font.render(f"{i}", True, (220, 220, 220))
				screen.blit(label, (ui_x + 20, by))
				# 背景栏
				pygame.draw.rect(screen, (80, 80, 80), (ui_x + 50, by, max_w, 16))
				# 概率栏
				pygame.draw.rect(screen, (100, 180, 240), (ui_x + 50, by, int(max_w * p), 16))
				pct = font.render(f"{p*100:5.1f}%", True, (220, 220, 220))
				screen.blit(pct, (ui_x + 50 + max_w + 6, by - 2))

		# 显示当前画笔灰度强度
		inten_label = font.render(f'强度: {brush_intensity}', True, (220, 220, 220))
		screen.blit(inten_label, (ui_x + 20, WIN_H - 70))
		# 显示灰度色块
		sw_x, sw_y, sw_w, sw_h = ui_x + 100, WIN_H - 78, 40, 24
		val = int(brush_intensity)
		pygame.draw.rect(screen, (val, val, val), (sw_x, sw_y, sw_w, sw_h))
		# 显示画笔半径
		radius_label = font.render(f'半径: {brush_radius}', True, (220, 220, 220))
		screen.blit(radius_label, (ui_x + 20, WIN_H - 100))

		# 显示说明
		hint = font.render('工具: 点击对应按钮切换，拖动绘制', True, (200, 200, 200))
		screen.blit(hint, (ui_x + 20, WIN_H - 30))

		pygame.display.flip()
		clock.tick(60)

	pygame.quit()

if __name__ == '__main__':
	run_pygame()

