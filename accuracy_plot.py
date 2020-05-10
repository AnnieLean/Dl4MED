# Accuracy Plot

import json
import matplotlib.pyplot as plt

inception_acc_dict = json.load(open("inceptionV3_acc_dict.txt"))
cnn_acc_dict = json.load(open("ConvNet_acc_dict.txt"))
inception_pretrained_acc_dict = json.load(open("inceptionV3_sampler_pretrained_acc_dict.txt")) 

fig = plt.figure(figsize=(8, 10))
ax1 = fig.add_subplot(3, 1, 1)
plt.plot(inception_acc_dict['train'], label='train')
plt.plot(inception_acc_dict['val'], label='valid')
plt.title('Accuracy of InceptionV3 per epoch (lr = 1e-03)')
plt.legend()

ax2 = fig.add_subplot(3, 1, 2) 
plt.plot(cnn_acc_dict['train'][:10], label='train')
plt.plot(cnn_acc_dict['val'][:10], label='valid')
plt.title('Accuracy of CNN per epoch (lr = 1e-03)')
plt.legend()

ax3 = fig.add_subplot(3, 1, 3) 
plt.plot(inception_pretrained_acc_dict['train'][:10], label='train')
plt.plot(inception_pretrained_acc_dict['val'][:10], label='valid')
plt.title('Accuracy of Pretrained InceptionV3 per epoch (lr = 1e-03)')
plt.legend()

plt.show()
fig.savefig('acc_plot.png')