from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 14,
    "figure.figsize": (10, 6),
    "lines.linewidth": 2.5,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# 加载 tfevent 文件
event_file = "logs_voc_frompretrain/events.out.tfevents.1748419432.1e7820eb3a37.31659.0"
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()

# 输出可用的 tag
# print("Available tags:", ea.Tags())

# 提取 train_loss 和 test_loss（根据实际 tag 名称替换）
train_loss_data = ea.Scalars("train_loss")
test_loss_data = ea.Scalars("test loss")

train_steps = [x.step for x in train_loss_data]
train_values = [x.value for x in train_loss_data]

test_steps = [x.step for x in test_loss_data]
test_values = [x.value for x in test_loss_data]

# 创建图形和左轴（训练损失）
fig, ax1 = plt.subplots()

color1 = "tab:blue"
ax1.plot(train_steps, train_values, label="Train Loss", color=color1)
ax1.set_xlabel("Step")
ax1.set_ylabel("Train Loss", color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

# 创建右轴（测试损失）
ax2 = ax1.twinx()
color2 = "tab:red"
ax2.plot(test_steps, test_values, label="Val Loss Classifier", color=color2, linestyle="--")
ax2.set_ylabel("Val Loss Classifier", color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

# 添加图例和标题
plt.title("Training and Test Loss Curve")
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=18)
fig.tight_layout()
plt.grid(True)

# 可选：保存图像
plt.savefig("train_test_loss_dual_axis.pdf", dpi=300, bbox_inches='tight')

plt.show()
