import matplotlib.pyplot as plt
import scienceplots
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2
# 设置字体为系统中存在的字体
plt.rcParams['font.family'] = 'serif'  # 使用 serif 字体族
plt.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Times']

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
	"""
	绘制时间序列值和异常分数随时间变化图
	Parameters
	----------
	name: model_dataset
	y_true: 时间序列
	y_pred: 重构的时间序列
	ascore：loss
	labels: 异常标记标签

	Returns
	-------

	"""
	# 将y_true也沿着0维度滚动1位，从而使y_true和y_pred时间对齐
	if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/output.pdf')
	for dim in range(y_true.shape[1]):
		# 按照时间序列的特征维度绘制
		y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
		fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		# 绘制真实的时间序列
		ax1.plot(smooth(y_t), linewidth=0.2, label='True')
		# 绘制重构的时间序列
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
		ax3 = ax1.twinx()
		# 绘制正常或异常标签
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		# 绘制Anomaly Score
		ax2.plot(smooth(a_s), linewidth=0.2, color='g')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		pdf.savefig(fig)
		plt.close()
	pdf.close()
