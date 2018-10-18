# -*-coding=UTF-8-*-
"""
在无参考图下，检测图片质量的方法
"""
import os
from pathlib import Path
import cv2

import numpy as np
from skimage import filters


class BlurDetection:
    def __init__(self, strDir):
        print("图片检测对象已经创建...")
        self.strDir = strDir

    def _getAllImg(self, strType='jpg'):
        """
        根据目录读取所有的图片
        :param strType: 图片的类型
        :return:  图片列表
        """
        names = []
        path = Path(self.strDir)
        for pa in path.glob('*.jpg'):
            names.append(pa.name)
        return names

    def _imageToMatrix(self, image):
        """
        根据名称读取图片对象转化矩阵
        :param strName:
        :return: 返回矩阵
        """
        imgMat = np.matrix(image)
        return imgMat

    def _blurDetection(self, imgName):

        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        imgMat=self._imageToMatrix(img2gray)/255.0
        x, y = imgMat.shape
        score = 0
        for i in range(x - 2):
            for j in range(y - 2):
                score += (imgMat[i + 2, j] - imgMat[i, j]) ** 2
        # step3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score=score/10
        newImg = self._drawImgFonts(reImg, str(score))
        newDir = self.strDir + "/_blurDetection_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        # newPath = newDir + imgName
        # cv2.imwrite(newPath, newImg)  # 保存图片
        # cv2.imshow(imgName, newImg)
        # cv2.waitKey(0)
        return score

    def _SMDDetection(self, imgName):

        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f=self._imageToMatrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])+np.abs(f[i,j]-f[i+1,j])
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score=score/100
        # newImg = self._drawImgFonts(reImg, str(score))
        # newDir = self.strDir + "/_SMDDetection_/"
        # if not os.path.exists(newDir):
        #     os.makedirs(newDir)
        # newPath = newDir + imgName
        # cv2.imwrite(newPath, newImg)  # 保存图片
        # cv2.imshow(imgName, newImg)
        # cv2.waitKey(0)
        return score

    def _SMD2Detection(self, imgName):
        """
        灰度方差乘积
        :param imgName:
        :return:
        """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f=self._imageToMatrix(img2gray)/255.0
        x, y = f.shape
        score = 0
        for i in range(x - 1):
            for j in range(y - 1):
                score += np.abs(f[i+1,j]-f[i,j])*np.abs(f[i,j]-f[i,j+1])
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score=score
        newImg = self._drawImgFonts(reImg, str(score))
        newDir = self.strDir + "/_SMD2Detection_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)
        return score
    def _Variance(self, imgName):
        """
               灰度方差乘积
               :param imgName:
               :return:
               """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f = self._imageToMatrix(img2gray)

        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        score = np.var(f)
        # newImg = self._drawImgFonts(reImg, str(score))
        # newDir = self.strDir + "/_Variance_/"
        # if not os.path.exists(newDir):
            # os.makedirs(newDir)
        # newPath = newDir + imgName
        # cv2.imwrite(newPath, newImg)  # 保存图片
        # cv2.imshow(imgName, newImg)
        # cv2.waitKey(0)
        return score
    def _Vollath(self,imgName):
        """
                       灰度方差乘积
                       :param imgName:
                       :return:
                       """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f = self._imageToMatrix(img2gray)
        source=0
        x,y=f.shape
        for i in range(x-1):
            for j in range(y):
                source+=f[i,j]*f[i+1,j]
        source=source-x*y*np.mean(f)
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分

        newImg = self._drawImgFonts(reImg, str(source))
        newDir = self.strDir + "/_Vollath_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)
        return source
    def _Tenengrad(self,imgName):
        """
                       灰度方差乘积
                       :param imgName:
                       :return:
                       """
        # step 1 图像的预处理
        img2gray, reImg = self.preImgOps(imgName)
        f = self._imageToMatrix(img2gray)

        tmp = filters.sobel(f)
        source=np.sum(tmp**2)
        source=np.sqrt(source)
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分

        newImg = self._drawImgFonts(reImg, str(source))
        newDir = self.strDir + "/_Tenengrad_/"
        if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        cv2.imwrite(newPath, newImg)  # 保存图片
        # cv2.imshow(imgName, newImg)
        # cv2.waitKey(0)
        return source

    def Test_Tenengrad(self):
        imgList = self._getAllImg(self.strDir)
        total = 0
        for i in range(len(imgList)):
            score = self._Tenengrad(imgList[i])
            total += score
            print(str(imgList[i]) + " is " + str(score))
        print(total / len(imgList))

    def Test_Vollath(self):
        imgList = self._getAllImg(self.strDir)
        total = 0
        for i in range(len(imgList)):
            score = self._Variance(imgList[i])
            total += score
            print(str(imgList[i]) + " is " + str(score))
        print(total / len(imgList))


    def TestVariance(self):
        imgList = self._getAllImg(self.strDir)
        total = 0
        for i in range(len(imgList)):
            score = self._Variance(imgList[i])
            total += score
            print(str(imgList[i]) + " is " + str(score))
        print(total / len(imgList))

    def TestSMD2(self):
        imgList = self._getAllImg(self.strDir)

        for i in range(len(imgList)):
            score = self._SMD2Detection(imgList[i])
            print(str(imgList[i]) + " is " + str(score))
        return
    def TestSMD(self):
        imgList = self._getAllImg(self.strDir)
        total = 0
        for i in range(len(imgList)):
            score = self._SMDDetection(imgList[i])
            total += score
            print(str(imgList[i]) + " is " + str(score))
        print(total / len(imgList))

    def TestBrener(self):
        imgList = self._getAllImg(self.strDir)
        total = 0
        for i in range(len(imgList)):
            score = self._blurDetection(imgList[i])
            total += score
            print(str(imgList[i]) + " is " + str(score))
        print(total / len(imgList))

    def preImgOps(self, imgName):
        """
        图像的预处理操作
        :param imgName: 图像的而明朝
        :return: 灰度化和resize之后的图片对象
        """
        strPath = self.strDir + imgName

        img = cv2.imread(strPath)  # 读取图片
        # cv2.moveWindow("", 1000, 100)
        # cv2.imshow("原始图", img)
        # 预处理操作
        reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  #
        img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图
        return img2gray, reImg

    def _drawImgFonts(self, img, strContent):
        """
        绘制图像
        :param img: cv下的图片对象
        :param strContent: 书写的图片内容
        :return:
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontSize = 5
        # 照片 添加的文字    /左上角坐标   字体   字体大小   颜色        字体粗细
        cv2.putText(img, strContent, (0, 200), font, fontSize, (0, 255, 0), 6)

        return img

    def _lapulaseDetection(self, imgName):
        """
        :param strdir: 文件所在的目录
        :param name: 文件名称
        :return: 检测模糊后的分数
        """
        # step1: 预处理
        img2gray, reImg = self.preImgOps(imgName)
        # step2: laplacian算子 获取评分
        resLap = cv2.Laplacian(img2gray, cv2.CV_64F)
        score = resLap.var()
        print("Laplacian %s score of given image is %s", str(score))
        # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
        # newImg = self._drawImgFonts(reImg, str(score))
        # newDir = self.strDir + "/_lapulaseDetection_/"
        # if not os.path.exists(newDir):
        #     os.makedirs(newDir)
        # newPath = newDir + imgName
        # # 显示
        # cv2.imwrite(newPath, newImg)  # 保存图片
        # cv2.imshow(imgName, newImg)
        # cv2.waitKey(0)

        # step3: 返回分数
        return score

    def TestDect(self):
        names = self._getAllImg()
        total = 0
        for i in range(len(names)):
            score = self._lapulaseDetection(names[i])
            total += score
            print(str(names[i]) + " is " + str(score))
        print(total / len(names))


if __name__ == "__main__":
    BlurDetection = BlurDetection(strDir="data/input/License/temp/Bad_License/")
    BlurDetection.TestVariance() # TestSMD