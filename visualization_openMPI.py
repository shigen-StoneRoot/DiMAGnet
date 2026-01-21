import matplotlib.pyplot as plt
import numpy as np



root_dir = r'./save_data'
num = 3

data = np.load(r'{}/samples_{}x1x64x64.npz'.format(root_dir, num))['arr_0'].squeeze()
img = data.squeeze()


plt.subplot(131)
plt.imshow(img[0], cmap='gray')
plt.axis('off')
plt.title('reco')

plt.subplot(132)
plt.imshow(img[1], cmap='gray')
plt.axis('off')
plt.title('GT')

plt.subplot(133)
plt.imshow(img[2], cmap='gray')
plt.axis('off')
plt.title('error')

plt.savefig("./visu.jpg")
plt.show()







