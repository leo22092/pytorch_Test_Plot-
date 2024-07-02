import matplotlib.pyplot as plt
import numpy as np

metrics = ['Accuracy','Precision', 'Recall', 'F1 Score']
AlexNet = [0.84, 0.8468, 0.8344, 0.8405]
VGG16 = [0.853,0.8701, 0.8428, 0.8562]
Resnet50 = [0.8938,0.8932, 0.8923, 0.8927]
Custom_Net = [0.81, 0.79, 0.78,0.7850]

x = np.arange(len(metrics))
width = 0.2

fig, ax = plt.subplots(figsize=(8, 6))
rects1 = ax.bar(x - 1.5*width, AlexNet, width, label='AlexNet')
rects2 = ax.bar(x - 0.5*width, VGG16, width, label='VGG16')
rects3 = ax.bar(x + 0.5*width, Resnet50, width, label='Resnet 50')
rects4 = ax.bar(x + 1.5*width, Custom_Net, width, label='Custom Net')

ax.set_xlabel('Metrics')
ax.set_xticks(x)
ax.set_xticklabels(metrics)

ax.legend(bbox_to_anchor=(0.5, -0.2), loc='lower center', ncol=4, fontsize=10)
plt.tight_layout()

plt.show()