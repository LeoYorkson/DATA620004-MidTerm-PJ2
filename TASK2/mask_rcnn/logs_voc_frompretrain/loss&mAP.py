# from tensorboard.backend.event_processing import event_accumulator
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 设置绘图风格和字体
# sns.set(style="whitegrid")
# plt.rcParams.update({
#     "font.family": "Times New Roman",
#     "font.size": 14,
#     "figure.figsize": (16, 6),  # 16宽，6高，方便两个子图横排
#     "lines.linewidth": 2.5,
#     "axes.titlesize": 16,
#     "axes.labelsize": 14,
#     "legend.fontsize": 12,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12
# })

# # 加载 tfevent 文件
# event_file = "logs_voc_frompretrain/events.out.tfevents.1748419432.1e7820eb3a37.31659.0"
# ea = event_accumulator.EventAccumulator(event_file)
# ea.Reload()

# # 输出可用 tags，方便确认
# print("Available tags:", ea.Tags())

# # 提取 train_loss 和 test_loss
# train_loss_data = ea.Scalars("train_loss")
# test_loss_data = ea.Scalars("test loss")

# train_steps = [x.step for x in train_loss_data]
# train_values = [x.value for x in train_loss_data]

# test_steps = [x.step for x in test_loss_data]
# test_values = [x.value for x in test_loss_data]

# # 提取 val_map
# val_map_data = ea.Scalars("mAP")  # 这里确保你的tag是val_map
# val_map_steps = [x.step for x in val_map_data]
# val_map_values = [x.value for x in val_map_data]

# # 创建左右两个子图
# fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

# ### 左图：train_loss 和 test_loss 双Y轴 ###
# color1 = "tab:blue"
# ax1.plot(train_steps, train_values, label="Train Loss", color=color1)
# ax1.set_xlabel("Step")
# ax1.set_ylabel("Train Loss", color=color1)
# ax1.tick_params(axis='y', labelcolor=color1)

# ax2 = ax1.twinx()
# color2 = "tab:red"
# ax2.plot(test_steps, test_values, label="Test Loss", color=color2, linestyle="--")
# ax2.set_ylabel("Test Loss", color=color2)
# ax2.tick_params(axis='y', labelcolor=color2)

# ax1.set_title("Training and Test Loss Curve")
# lines_1, labels_1 = ax1.get_legend_handles_labels()
# lines_2, labels_2 = ax2.get_legend_handles_labels()
# ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
# ax1.grid(True)

# ### 右图：val_map 曲线 ###
# ax3.plot(val_map_steps, val_map_values, color="tab:cyan", label="Validation mAP")
# ax3.set_xlabel("Step")
# ax3.set_ylabel("mAP")
# ax3.set_title("Validation mAP Curve")
# ax3.legend(loc="lower right")
# ax3.grid(True)

# plt.tight_layout()

# plt.savefig("loss_and_mAP.pdf", dpi=300, bbox_inches='tight')
# plt.show()

from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.2, context="talk", color_codes=False)
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# 加载 tfevent 文件
# event_file = "logs_voc_frompretrain/events.out.tfevents.1748419432.1e7820eb3a37.31659.0"
# event_file = 'logs_voc_frompretrain\events.out.tfevents.1748511638.1e7820eb3a37.263213.0'
event_file= 'logs_voc_frompretrain\events.out.tfevents.1748519030.1e7820eb3a37.274695.0'
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# 提取数据
train_loss_data = ea.Scalars("train_loss")
test_loss_data = ea.Scalars("val loss")
val_map_data = ea.Scalars("mAP")

train_steps = [x.step for x in train_loss_data]
train_values = [x.value for x in train_loss_data]
test_steps = [x.step for x in test_loss_data]
test_values = [x.value for x in test_loss_data]
val_map_steps = [x.step for x in val_map_data]
val_map_values = [x.value for x in val_map_data]

plt.rcParams['font.family'] = 'Times New Roman'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

### 左子图：train_loss 和 test_loss ###
# 整理绘图数据和样式
loss_plot_data = [
    (train_steps, train_values, 'Train Loss', '-', 'tab:blue'),
    (test_steps, test_values, 'Test Loss', '--', 'tab:red')
]

# 先在左轴画 train_loss
for x, y, label, linestyle, color in loss_plot_data:
    if 'Train' in label:
        ax1.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=3)
ax1.set_xlabel('Step', fontweight='bold', fontsize=16)
ax1.set_ylabel('Train Loss', fontweight='bold', fontsize=16, color='tab:blue')
ax1.tick_params(axis='y', colors='tab:blue', labelsize=14, width=2)
ax1.tick_params(axis='x', labelsize=14, width=2)
for tick in ax1.xaxis.get_major_ticks() + ax1.yaxis.get_major_ticks():
    # tick.label.set_fontweight('bold')
    tick_label = tick.label1 if hasattr(tick, "label1") else tick.label
    tick_label.set_fontweight('bold')


# 右轴画 test_loss
ax1_r = ax1.twinx()
for x, y, label, linestyle, color in loss_plot_data:
    if 'Test' in label:
        ax1_r.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=3)
ax1_r.set_ylabel('Test Loss', fontweight='bold', fontsize=16, color='tab:red')
ax1_r.tick_params(axis='y', colors='tab:red', labelsize=14, width=2)
for tick in ax1_r.yaxis.get_major_ticks():
    # tick.label.set_fontweight('bold')
    tick_label = tick.label1 if hasattr(tick, "label1") else tick.label
    tick_label.set_fontweight('bold')


ax1.set_title('Training and Test Loss Curve', fontweight='bold', fontsize=18)

# 合并图例
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1_r.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', prop={'weight': 'bold', 'size':14})

ax1.grid(True)

### 右子图：val_map ###
map_plot_data = [
    (val_map_steps, val_map_values, 'Validation mAP', '-', 'tab:green')
]

for x, y, label, linestyle, color in map_plot_data:
    ax2.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=3)

ax2.set_xlabel('Step', fontweight='bold', fontsize=16)
ax2.set_ylabel('mAP', fontweight='bold', fontsize=16)
ax2.tick_params(axis='both', labelsize=14, width=2)
for tick in ax2.xaxis.get_major_ticks() + ax2.yaxis.get_major_ticks():
    # tick.label.set_fontweight('bold')
    tick_label = tick.label1 if hasattr(tick, "label1") else tick.label
    tick_label.set_fontweight('bold')

ax2.set_title('Validation mAP Curve', fontweight='bold', fontsize=18)
ax2.legend(prop={'weight': 'bold', 'size':14}, loc='lower right')
ax2.grid(True)

plt.tight_layout()
plt.savefig("loss_and_mAP.pdf", dpi=300, bbox_inches='tight')
plt.show()
