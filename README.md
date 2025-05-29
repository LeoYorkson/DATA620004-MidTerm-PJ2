# DATA620004-MidTerm-PJ2 深度学习实验报告：图像分类与实例分割

## 📌 实验任务简介

本项目包含两个任务：

### 任务一：在 Caltech-101 数据集上微调预训练 CNN 模型进行图像分类

- 使用在 ImageNet 上预训练的卷积神经网络（如 ResNet-18、AlexNet）；
- 修改输出层以适配 Caltech-101 的 101 个类别；
- 对新输出层进行训练，同时以较小学习率微调预训练部分；
- 比较预训练与从零开始训练的性能差异。

### 任务二：在 Pascal VOC 数据集上训练并测试实例分割模型 Mask R-CNN 和 Sparse R-CNN

- 在 VOC 数据集上训练 Mask R-CNN 和 Sparse R-CNN 模型；
- 对模型在测试图像上的目标检测与实例分割结果进行可视化；
- 对比 proposal 阶段与最终预测的差异；
- 分析模型在非 VOC 图像上的泛化能力。

---

## 🧠 模型简述

- **任务一模型**：使用 ImageNet 预训练的以及从头开始训练的 ResNet-18，替换输出层以适应 Caltech-101 分类任务；
- **任务二模型**：采用经典的 Mask R-CNN（两阶段、基于 proposal）与 Sparse R-CNN（基于 query 的端到端检测）进行目标检测与实例分割。

---

## 📊 实验结果摘要

- ✅ **任务一**：完成了模型的训练与微调，使用 TensorBoard 记录了训练集与验证集上的 loss 曲线及准确率变化。对比实验验证了预训练模型的显著优势；

  <table>
    <tr>
      <td>
        <a href="https://postimg.cc/Yvcqr2sx" target="_blank">
          <img src="https://i.postimg.cc/7ZhCNf1j/1-image.png" alt="Caltech图像" width="450"/>
        </a>
      </td>
      <td>
        <a href="https://postimg.cc/qt3Fk0W2" target="_blank">
          <img src="https://i.postimg.cc/tgSGN45v/Screenshot-2025-05-29-223853.png" alt="任务一结果" width="450"/>
        </a>
      </td>
    </tr>
  </table>
  
- ✅ **任务二**：基于现有框架成功完成了 VOC 上的 Mask R-CNN 与 Sparse R-CNN 训练，并对比展示了 proposal 与最终预测的差异、两个模型的检测与分割效果，以及在 VOC 外部图像上的表现。

  <table>
    <tr>
      <td>
        <a href="https://postimg.cc/ctH6nDRM" target="_blank">
          <img src="https://i.postimg.cc/J03XSvtg/loss-train.png" alt="训练 loss 曲线" width="450"/>
        </a>
      </td>
      <td>
        <a href="https://postimg.cc/zby9HKLH" target="_blank">
          <img src="https://i.postimg.cc/DZ6FDcn6/cat-person.jpg" alt="VOC 外部检测图" width="450"/>
        </a>
      </td>
    </tr>
  </table>


## 🛠️ 使用说明

请分别进入任务子文件夹查看对应的训练与测试方法说明：

- 📁 [`TASK1/`](./TASK1)：包含 Caltech-101 图像分类任务的代码、训练说明及模型权重；
- 📁 [`TASK2/`](./TASK2)：包含 VOC 实例分割任务（Mask R-CNN 和 Sparse R-CNN）的训练、可视化、评估代码与说明。

---

## 🔗 附件链接

- 📄 实验报告（PDF）：[链接待添加]
- 📦 模型权重下载地址（ Google Drive）：[https://drive.google.com/drive/folders/1nfiKODUAGKucymPrm--stA-me-QFljkR?usp=sharing]

