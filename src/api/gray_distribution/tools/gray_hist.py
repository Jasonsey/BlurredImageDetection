from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

for path in Path('data/input/License/temp/Bad_License/').glob('*.jpg'):
    # 打开图像，并转化成灰度图像
    image = Image.open(path).convert("L") 
    image_array = np.array(image)
    print(image_array.shape)

    plt.cla()
    plt.subplot(2,1,1)
    plt.imshow(image,cmap=cm.gray)
    plt.axis("off")
    plt.subplot(2,1,2)
    plt.hist(image_array.flatten(),256) #flatten可以将矩阵转化成一维序列
    plt.show()