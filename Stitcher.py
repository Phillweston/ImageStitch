import numpy as np
import cv2
import time
import os
import glob
import copy
import ImageUtility as Utility
import ImageFusion
import time


class ImageFeature():
    # 用来保存串行全局拼接中的第二张图像的特征点和描述子，为后续加速拼接使用，避免重复计算
    isBreak = True      # 判断是否上一次中断
    kps = None
    feature = None


class Stitcher(Utility.Method):
    '''
	    图像拼接类，包括所有跟材料显微组织图像配准相关函数
	'''
    isColorMode = True
    windowing = False
    direction = 1               # 1： 第一张图像在上，第二张图像在下；   2： 第一张图像在左，第二张图像在右；
                                # 3： 第一张图像在下，第二张图像在上；   4： 第一张图像在右，第二张图像在左；
    directIncre = 1             # 拼接增长方向，可以为1. 0， -1
    fuseMethod = "notFuse"
    phaseResponseThreshold = 0.15
    tempImageFeature = ImageFeature()

    imageFusion = ImageFusion.ImageFusion()


    def directionIncrease(self, direction):
        """
        功能：改变拼接搜索方向，通过direction和directIncre控制，使得范围保持在[1,4]
        :param direction: 当前的方向
        :return: 返回更新后的方向
        """
        direction = (direction - 1 + self.directIncre) % 4 + 1
        return direction

    def flowStitch(self, fileList, calculateOffsetMethod):
        """
        功能：序列拼接，从list的第一张拼接到最后一张，由于中间可能出现拼接失败，故记录截止文件索引
        :param fileList: 图像地址序列
        :param calculateOffsetMethod:计算偏移量方法
        :return: ((status, endfileIndex), stitchImage),（（拼接状态， 截止文件索引）， 拼接结果）
        """
        self.printAndWrite("Stitching the directory which have " + str(fileList[0]))
        fileNum = len(fileList)
        offsetList = []
        # calculating the offset for small image
        startTime = time.time()
        status = True
        endfileIndex = 0

        for fileIndex in range(0, fileNum - 1):
            self.printAndWrite("stitching " + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex + 1]))
            # imageA = cv2.imread(fileList[fileIndex], 0)
            # imageB = cv2.imread(fileList[fileIndex + 1], 0)
            imageA = cv2.imdecode(np.fromfile(fileList[fileIndex], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            imageB = cv2.imdecode(np.fromfile(fileList[fileIndex + 1], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if calculateOffsetMethod == self.calculateOffsetForPhaseCorrelate:
                (status, offset) = self.calculateOffsetForPhaseCorrelate([fileList[fileIndex], fileList[fileIndex + 1]])
            else:
                (status, offset) = calculateOffsetMethod([imageA, imageB])
            if not status:
                self.printAndWrite("  " + str(fileList[fileIndex]) + " and " + str(fileList[fileIndex+1]) + " cannot be stitched")
                break
            else:
                offsetList.append(offset)
                endfileIndex = fileIndex + 1
        endTime = time.time()

        self.printAndWrite("The time of registering is " + str(endTime - startTime) + "s")

        # stitching and fusing
        self.printAndWrite("start stitching")
        startTime = time.time()
        # offsetList = [[1784, 2], [1805, 2], [1809, 2], [1775, 2], [1760, 2], [1846, 2], [1809, 1], [1812, 2], [1786, 1], [1818, 3], [1786, 2], [1802, 2], [1722, 1], [1211, 1], [-10, 2411], [-1734, -1], [-1808, -1], [-1788, -3], [-1754, -1], [-1727, -2], [-1790, -3], [-1785, -2], [-1778, -1], [-1807, -2], [-1767, -2], [-1822, -3], [-1677, -2], [-1778, -2], [-1440, -1], [-2, 2410], [1758, 2], [1792, 2], [1794, 2], [1840, 3], [1782, 2], [1802, 3], [1782, 2], [1763, 3], [1738, 2], [1837, 3], [1781, 2], [1788, 18], [1712, 0], [1271, -11], [-3, 2478], [-1787, -1], [-1812, -2], [-1822, -2], [-1762, -1], [-1725, -2], [-1884, -2], [-1754, -2], [-1747, -1], [-1666, -1], [-1874, -3], [-1695, -2], [-1672, -1], [-1816, -2], [-1411, -1], [-4, 2431], [1874, 3], [1706, -3], [1782, 2], [1794, 2], [1732, 3], [1838, 3], [1721, 1], [1783, 3], [1805, 2], [1725, 3], [1828, 1], [1774, 3], [1776, 1], [1201, 1], [-16, 2405], [-1821, 0], [-1843, -2], [-1758, -2], [-1742, -3], [-1814, -2], [-1817, -2], [-1848, -2], [-1768, -2], [-1749, -2], [-1765, -2], [-1659, -2], [-1832, -2], [-1791, -2], [-1197, -1]]
        stitchImage = self.getStitchByOffset(fileList, offsetList)
        endTime = time.time()
        self.printAndWrite("The time of fusing is " + str(endTime - startTime) + "s")

        return ((status, endfileIndex), stitchImage)

    def flowStitchWithMultiple(self, fileList, calculateOffsetMethod):
        """
        功能：多段序列拼接，从list的第一张拼接到最后一张，由于中间可能出现拼接失败，将分段拼接结果共同返回
        :param fileList: 图像地址序列
        :param calculateOffsetMethod:计算偏移量方法
        :return: 拼接的图像list
        """
        result = []
        totalNum = len(fileList)
        startNum = 0
        while True:
            (status, stitchResult) = self.flowStitch(fileList[startNum: totalNum], calculateOffsetMethod)
            result.append(stitchResult)
            self.tempImageFeature.isBreak = True
            startNum += status[1] + 1

            # self.printAndWrite("status[1] = " + str(status[1]))
            # self.printAndWrite("startNum = "+str(startNum))
            if startNum >= totalNum - 1:
                if startNum == totalNum - 1:  # Handle the last image if not included in stitching
                    mode = cv2.IMREAD_COLOR if self.isColorMode else cv2.IMREAD_GRAYSCALE
                    result.append(cv2.imdecode(np.fromfile(fileList[startNum], dtype=np.uint8), mode))
                break
            self.printAndWrite("stitching Break, start from " + str(fileList[startNum]) + " again")
        return result

    def imageSetStitch(self, projectAddress, outputAddress, fileNum, calculateOffsetMethod, startNum = 1, fileExtension = "jpg", outputfileExtension = "jpg"):
        """
        功能：图像集拼接方法
        :param projectAddress: 项目地址
        :param outputAddress: 输出地址
        :param fileNum: 共多少个文件
        :param calculateOffsetMethod: 计算偏移量方法
        :param startNum: 从第几个文件开始拼
        :param fileExtension: 输入文件扩展名
        :param outputfileExtension:输出文件扩展名
        :return:
        """
        os.makedirs(outputAddress, exist_ok=True)  # Ensure output directory exists
        for i in range(startNum, fileNum+1):
            fileAddress = projectAddress + "\\" + str(i) + "\\"
            fileList = glob.glob(fileAddress + "*." + fileExtension)
            if not fileList:  # Check if fileList is empty
                self.printAndWrite(f"No files found in {fileAddress} with extension {fileExtension}. Skipping...")
                continue
            (status, result) = self.flowStitch(fileList, calculateOffsetMethod)
            self.tempImageFeature.isBreak = True
            cv2.imwrite(outputAddress + "\\stitching_result_" + str(i) + "." + outputfileExtension, result)
            if status == False:
                self.printAndWrite("stitching Failed")

    def imageSetStitchWithMultiple(self, projectAddress, outputAddress, fileNum, calculateOffsetMethod, startNum = 1, fileExtension = "jpg", outputfileExtension = "jpg"):
        """
        功能：图像集多段拼接方法
        :param projectAddress: 项目地址
        :param outputAddress: 输出地址
        :param fileNum: 共多少个文件
        :param calculateOffsetMethod: 计算偏移量方法
        :param startNum: 从第几个文件开始拼
        :param fileExtension: 输入文件扩展名
        :param outputfileExtension:输出文件扩展名
        :return:
        """
        os.makedirs(outputAddress, exist_ok=True)  # Ensure output directory exists
        for i in range(startNum, fileNum+1):
            startTime = time.time()
            fileAddress = projectAddress + "\\" + str(i) + "\\"
            fileList = glob.glob(fileAddress + "*." + fileExtension)
            if not fileList:  # Check if fileList is empty
                self.printAndWrite(f"No files found in {fileAddress} with extension {fileExtension}. Skipping...")
                continue
            result = self.flowStitchWithMultiple(fileList, calculateOffsetMethod)
            self.tempImageFeature.isBreak = True
            if len(result) == 1:
                cv2.imwrite(outputAddress + "\\stitching_result_" + str(i) + "." + outputfileExtension, result[0])
                # cv2.imwrite(outputAddress + "\\" + outputName + "." + outputfileExtension, result[0])
            else:
                for j in range(0, len(result)):
                    cv2.imwrite(outputAddress + "\\stitching_result_" + str(i) + "_" + str(j+1) + "." + outputfileExtension, result[j])
                    # cv2.imwrite(outputAddress + "\\" + outputName + "_" + str(j + 1) + "." + outputfileExtension,result[j])
            endTime = time.time()
            self.printAndWrite("Time Consuming for " + fileAddress + " is " + str(endTime - startTime))

    def calculateOffsetForPhaseCorrelate(self, dirAddress):
        """
        功能：采用相位相关法计算偏移量（不完善）
        :param dirAddress: 图像文件夹地址
        :return: (status, offset)
        """
        (dir1, dir2) = dirAddress
        offsetList = self.phase.phaseCorrelation(dir1, dir2)
        # Convert offset list to integers and swap the order to match the expected output
        offset = [int(np.round(offsetList[1])), int(np.round(offsetList[0]))]
        self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
        return (True, offset)

    def calculateOffsetForPhaseCorrelateIncre(self, images):
        '''
        功能：采用相位相关法计算偏移量-考虑增长搜索区域（不完善）
        :param images: [imageA, imageB]
        :return: (status, offset)
        '''
        (imageA, imageB) = images
        offset = [0, 0]
        status = False
        maxI = (np.floor(0.5 / self.roiRatio) + 1).astype(int)+ 1
        iniDirection = self.direction
        localDirection = iniDirection
        for i in range(1, maxI):
            # self.printAndWrite("  i=" + str(i) + " and maxI="+str(maxI))
            while True:
                # get the roi region of images
                # self.printAndWrite("  localDirection=" + str(localDirection))
                roiImageA = self.getROIRegionForIncreMethod(imageA, direction=localDirection, order="first", searchRatio = i * self.roiRatio)
                roiImageB = self.getROIRegionForIncreMethod(imageB, direction=localDirection, order="second", searchRatio = i * self.roiRatio)

                if self.windowing:
                    hann = cv2.createHanningWindow(winSize=(roiImageA.shape[1], roiImageA.shape[0]), type=5)
                    (offsetTemp, response) = cv2.phaseCorrelate(np.float32(roiImageA), np.float32(roiImageB), window=hann)
                else:
                    (offsetTemp, response) = cv2.phaseCorrelate(np.float64(roiImageA), np.float64(roiImageB))

                offset = [int(offsetTemp[1]), int(offsetTemp[0])]
                # self.printAndWrite("offset: " + str(offset))
                # self.printAndWrite("respnse: " + str(response))
                if response > self.phaseResponseThreshold:
                    status = True
                    break
                else:
                    localDirection = self.directionIncrease(localDirection)
                    if localDirection == iniDirection:
                        break
            if status:
                if localDirection == 1:
                    offset[0] += imageA.shape[0] - int(i * self.roiRatio * imageA.shape[0])
                elif localDirection == 2:
                    offset[1] += imageA.shape[1] - int(i * self.roiRatio * imageA.shape[1])
                elif localDirection == 3:
                    offset[0] -= imageB.shape[0] - int(i * self.roiRatio * imageB.shape[0])
                elif localDirection == 4:
                    offset[1] -= imageB.shape[1] - int(i * self.roiRatio * imageB.shape[1])
                self.direction = localDirection
                break
        if not status:
            return (status, "  The two images can not match")

        self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
        return (status, offset)

    def calculateOffsetForFeatureSearch(self, images):
        '''
        功能：采用特征搜索计算偏移量
        :param images: [imageA, imageB]
        :return: (status, offset)
        '''
        (imageA, imageB) = images
        offset = [0, 0]
        status = False
        # Image enhancement
        if self.isEnhance:
            clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.tileSize, self.tileSize)) if self.isClahe else None
            for i, image in enumerate([imageA, imageB]):
                images[i] = clahe.apply(image) if clahe else cv2.equalizeHist(image)

        # Feature detection and description
        if self.tempImageFeature.isBreak:
            (kpsA, featuresA) = self.detectAndDescribe(imageA, featureMethod=self.featureMethod)
        else:
            kpsA, featuresA = self.tempImageFeature.kps, self.tempImageFeature.feature
        (kpsB, featuresB) = self.detectAndDescribe(imageB, featureMethod=self.featureMethod)

        # Update temporary image features
        self.tempImageFeature.isBreak = False
        self.tempImageFeature.kps, self.tempImageFeature.feature = kpsB, featuresB

        # Feature matching
        if featuresA is not None and featuresB is not None:
            matches = self.matchDescriptors(featuresA, featuresB)
            if self.offsetCalculate == "mode":
                (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate=self.offsetEvaluate)
            elif self.offsetCalculate == "ransac":
                (status, offset, _) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate=self.offsetEvaluate)

        # Handling the status
        if not status:
            self.tempImageFeature.isBreak = True
            return (status, "The two images cannot match")

        self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
        return (status, offset)

    def calculateOffsetForFeatureSearchIncre(self, images):
        '''
        功能：采用特征搜索计算偏移量-考虑增长搜索区域
        :param images: [imageA, imageB]
        :return: (status, offset)
        '''

        (imageA, imageB) = images
        offset = [0, 0]
        status = False
        maxI = (np.floor(0.5 / self.roiRatio) + 1).astype(int)+ 1
        iniDirection = self.direction
        localDirection = iniDirection
        for i in range(1, maxI):
            # self.printAndWrite("  i=" + str(i) + " and maxI="+str(maxI))
            while True:
                # get the roi region of images
                # self.printAndWrite("  localDirection=" + str(localDirection))
                roiImageA = self.getROIRegionForIncreMethod(imageA, direction=localDirection, order="first", searchRatio = i * self.roiRatio)
                roiImageB = self.getROIRegionForIncreMethod(imageB, direction=localDirection, order="second", searchRatio = i * self.roiRatio)

                if self.isEnhance:
                    clahe = cv2.createCLAHE(clipLimit=self.clipLimit, tileGridSize=(self.tileSize, self.tileSize)) if self.isClahe else None
                    for roiImage in [roiImageA, roiImageB]:
                        if self.isClahe:
                            roiImage[:] = clahe.apply(roiImage)
                        else:
                            roiImage[:] = cv2.equalizeHist(roiImage)
                # get the feature points
                kpsA, featuresA = self.detectAndDescribe(roiImageA, featureMethod=self.featureMethod)
                kpsB, featuresB = self.detectAndDescribe(roiImageB, featureMethod=self.featureMethod)
                if featuresA is not None and featuresB is not None:
                    matches = self.matchDescriptors(featuresA, featuresB)
                    # self.printAndWrite("  The number of raw matches is " + str(len(matches)))
                    # match all the feature points
                    if self.offsetCalculate == "mode":
                        (status, offset) = self.getOffsetByMode(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
                    elif self.offsetCalculate == "ransac":
                        (status, offset, _) = self.getOffsetByRansac(kpsA, kpsB, matches, offsetEvaluate = self.offsetEvaluate)
                if status or localDirection == iniDirection:
                    break
                localDirection = self.directionIncrease(localDirection)

            if status:
                adjustment = int(i * self.roiRatio * imageA.shape[0 if localDirection in [1, 3] else 1])
                if localDirection == 1:
                    offset[0] += imageA.shape[0] - adjustment
                elif localDirection == 2:
                    offset[1] += imageA.shape[1] - adjustment
                elif localDirection == 3:
                    offset[0] -= imageB.shape[0] - adjustment
                elif localDirection == 4:
                    offset[1] -= imageB.shape[1] - adjustment
                self.direction = localDirection
                break

        if not status:
            return (status, "  The two images cannot match")
        self.printAndWrite("  The offset of stitching: dx is " + str(offset[0]) + " dy is " + str(offset[1]))
        return (status, offset)

    def getStitchByOffset(self, fileList, originOffsetList):
        '''
        功能：通过偏移量列表和文件列表得到最终的拼接结果
        :param fileList: 图像列表
        :param originOffsetList: 偏移量列表
        :return: ndarry，图像
        '''
        # 如果你不细心，不要碰这段代码
        # 已优化到根据指针来控制拼接，CPU下最快了
        dxSum = dySum = 0
        imageList = []
        # imageList.append(cv2.imread(fileList[0], 0))
        if self.isColorMode:
            imageList.append(cv2.imdecode(np.fromfile(fileList[0], dtype=np.uint8), cv2.IMREAD_COLOR))
        else:
            imageList.append(cv2.imdecode(np.fromfile(fileList[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE))
        resultRow = imageList[0].shape[0]         # 拼接最终结果的横轴长度,先赋值第一个图像的横轴
        resultCol = imageList[0].shape[1]         # 拼接最终结果的纵轴长度,先赋值第一个图像的纵轴
        originOffsetList.insert(0, [0, 0])        # 增加第一张图像相对于最终结果的原点的偏移量

        rangeX = [[0, 0] for x in range(len(originOffsetList))]  # 主要用于记录X方向最大最小边界
        rangeY = [[0, 0] for x in range(len(originOffsetList))]  # 主要用于记录Y方向最大最小边界
        # print("originOffsetList=",originOffsetList)
        offsetList = copy.deepcopy(originOffsetList)
        rangeX[0][1] = imageList[0].shape[0]
        rangeY[0][1] = imageList[0].shape[1]

        for i in range(1, len(offsetList)):
            # self.printAndWrite("  stitching " + str(fileList[i]))
            # 适用于流形拼接的校正,并更新最终图像大小
            # tempImage = cv2.imread(fileList[i], 0)
            if Stitcher.isColorMode:
                tempImage = cv2.imdecode(np.fromfile(fileList[i], dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                tempImage = cv2.imdecode(np.fromfile(fileList[i], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            dxSum = dxSum + offsetList[i][0]
            dySum = dySum + offsetList[i][1]
            # self.printAndWrite("  The dxSum is " + str(dxSum) + " and the dySum is " + str(dySum))
            if dxSum <= 0:
                for j in range(0, i):
                    offsetList[j][0] = offsetList[j][0] + abs(dxSum)
                    rangeX[j][0] = rangeX[j][0] + abs(dxSum)
                    rangeX[j][1] = rangeX[j][1] + abs(dxSum)
                resultRow = resultRow + abs(dxSum)
                rangeX[i][1] = resultRow
                dxSum = rangeX[i][0] = offsetList[i][0] = 0
            else:
                offsetList[i][0] = dxSum
                resultRow = max(resultRow, dxSum + tempImage.shape[0])
                rangeX[i][1] = resultRow
            if dySum <= 0:
                for j in range(0, i):
                    offsetList[j][1] = offsetList[j][1] + abs(dySum)
                    rangeY[j][0] = rangeY[j][0] + abs(dySum)
                    rangeY[j][1] = rangeY[j][1] + abs(dySum)
                resultCol = resultCol + abs(dySum)
                rangeY[i][1] = resultCol
                dySum = rangeY[i][0] = offsetList[i][1] = 0
            else:
                offsetList[i][1] = dySum
                resultCol = max(resultCol, dySum + tempImage.shape[1])
                rangeY[i][1] = resultCol
            imageList.append(tempImage)
        stitchResult = None
        if self.isColorMode:
            stitchResult = np.zeros((resultRow, resultCol, 3), np.int32) - 1
        else:
            stitchResult = np.zeros((resultRow, resultCol), np.int32) - 1
        # stitchResult = np.zeros((resultRow, resultCol), np.int32)
        self.printAndWrite("  The rectified offsetList is " + str(offsetList))
        # 如上算出各个图像相对于原点偏移量，并最终计算出输出图像大小，并构造矩阵，如下开始赋值
        for i in range(0, len(offsetList)):
            self.printAndWrite("  stitching " + str(fileList[i]))
            if i == 0:
                if self.isColorMode:
                    stitchResult[offsetList[0][0]: offsetList[0][0] + imageList[0].shape[0], offsetList[0][1]: offsetList[0][1] + imageList[0].shape[1], :] = imageList[0]
                else:
                    stitchResult[offsetList[0][0]: offsetList[0][0] + imageList[0].shape[0], offsetList[0][1]: offsetList[0][1] + imageList[0].shape[1]] = imageList[0]
            else:
                if self.fuseMethod == "notFuse":
                    # 适用于无图像融合，直接覆盖
                    # self.printAndWrite("Stitch " + str(i+1) + "th, the roi_ltx is " + str(offsetList[i][0]) + " and the roi_lty is " + str(offsetList[i][1]))
                    if self.isColorMode:
                        stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0], offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1], :] = imageList[i]
                    else:
                        stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0], offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1]] = imageList[i]
                else:
                    # 适用于图像融合算法，切出 roiA 和 roiB 供图像融合
                    roi_ltx = max(offsetList[i][0], rangeX[i-1][0])
                    roi_lty = max(offsetList[i][1], rangeY[i-1][0])
                    roi_rbx = min(offsetList[i][0] + imageList[i].shape[0], rangeX[i-1][1])
                    roi_rby = min(offsetList[i][1] + imageList[i].shape[1], rangeY[i-1][1])
                    # self.printAndWrite("Stitch " + str(i + 1) + "th, the roi_ltx is " + str(
                    #     roi_ltx) + " and the roi_lty is " + str(roi_lty) + " and the roi_rbx is " + str(
                    #     roi_rbx) + " and the roi_rby is " + str(roi_rby))

                    if self.isColorMode:
                        roiImageRegionA = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby, :].copy()
                        stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0], offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1], :] = imageList[i]
                        roiImageRegionB = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby, :].copy()
                        stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby, :] = self.fuseImage([roiImageRegionA, roiImageRegionB], originOffsetList[i][0], originOffsetList[i][1])
                    else:
                        roiImageRegionA = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                        stitchResult[offsetList[i][0]: offsetList[i][0] + imageList[i].shape[0], offsetList[i][1]: offsetList[i][1] + imageList[i].shape[1]] = imageList[i]
                        roiImageRegionB = stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby].copy()
                        stitchResult[roi_ltx:roi_rbx, roi_lty:roi_rby] = self.fuseImage([roiImageRegionA, roiImageRegionB], originOffsetList[i][0], originOffsetList[i][1])
        # print("originOffsetList=", originOffsetList)
        stitchResult[stitchResult == -1] = 0
        return stitchResult.astype(np.uint8)

    def fuseImage(self, images, dx, dy):
        """
        功能：融合图像
        :param images: [imageA, imageB]
        :param dx: x方向偏移量
        :param dy: y方向偏移量
        :return: 融合后的图像
        """
        self.imageFusion.isColorMode = self.isColorMode
        (imageA, imageB) = images
        if self.fuseMethod not in ["fadeInAndFadeOut", "trigonometric"]:
            # 将各自区域中为背景的部分用另一区域填充，目的是消除背景
            # 权值为-1是为了方便渐入检出融合和三角融合计算
            imageA[imageA == -1] = 0
            imageB[imageB == -1] = 0
            imageA[imageA == 0] = imageB[imageA == 0]
            imageB[imageB == 0] = imageA[imageB == 0]

        fuseRegion = np.zeros(imageA.shape, np.uint8)
        if self.fuseMethod == "notFuse":
            fuseRegion = imageB
        elif self.fuseMethod == "average":
            fuseRegion = self.imageFusion.fuseByAverage([imageA, imageB])
        elif self.fuseMethod == "maximum":
            fuseRegion = self.imageFusion.fuseByMaximum([imageA, imageB])
        elif self.fuseMethod == "minimum":
            fuseRegion = self.imageFusion.fuseByMinimum([imageA, imageB])
        elif self.fuseMethod == "fadeInAndFadeOut":
            fuseRegion = self.imageFusion.fuseByFadeInAndFadeOut(images, dx, dy)
        elif self.fuseMethod == "trigonometric":
            fuseRegion = self.imageFusion.fuseByTrigonometric(images, dx, dy)
        elif self.fuseMethod == "multiBandBlending":
            assert self.isColorMode is False, "The multi Band Blending is not support for color mode in this code"
            fuseRegion = self.imageFusion.fuseByMultiBandBlending([imageA, imageB])
        elif self.fuseMethod == "optimalSeamLine":
            assert self.isColorMode is False, "The optimal seam line is not support for color mode in this code"
            fuseRegion = self.imageFusion.fuseByOptimalSeamLine(images, self.direction)
        return fuseRegion

if __name__=="__main__":
    stitcher = Stitcher()
    imageA = cv2.imread(".\\images\\dendriticCrystal\\1\\1-044.jpg", 0)
    imageB = cv2.imread(".\\images\\dendriticCrystal\\1\\1-045.jpg", 0)
    offset = stitcher.calculateOffsetForFeatureSearchIncre([imageA, imageB])