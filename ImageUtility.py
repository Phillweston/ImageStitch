import numpy as np
import cv2
import math
from cv2 import cuda


class Method():
    # 关于打印信息的设置
    outputAddress = "result/"
    isEvaluate = False          # 是否输出到检验txt文件
    evaluateFile = "evaluate.txt"
    isPrintLog = True           # 是否在屏幕打印过程信息

    # 关于特征搜索的设置
    featureMethod = "surf"      # "sift","surf" or "orb"
    roiRatio = 0.1              # roi length for stitching in first direction
    searchRatio = 0.75          # 0.75 is common value for matches

    # 关于 GPU 加速的设置
    isGPUAvailable = False       # 判断GPU目前是否可用

    # 关于 GPU-SURF 的设置
    surfHessianThreshold = 100.0
    surfNOctaves = 4
    surfNOctaveLayers = 3
    surfIsExtended = True
    surfKeypointsRatio = 0.01
    surfIsUpright = False

    # 关于 GPU-ORB 的设置
    orbNfeatures = 5000
    orbScaleFactor = 1.2
    orbNlevels = 8
    orbEdgeThreshold = 31
    orbFirstLevel = 0
    orbWTA_K = 2
    orbPatchSize = 31
    orbFastThreshold = 20
    orbBlurForDescriptor = False
    orbMaxDistance = 30

    # 关于特征配准的设置
    offsetCalculate = "mode"     # "mode" or "ransac"
    offsetEvaluate = 3           # 40 menas nums of matches for mode, 3.0 menas  of matches for ransac

    # 关于图像增强的操作
    isEnhance = False
    isClahe = False
    clipLimit = 20
    tileSize = 5

    def printAndWrite(self, content):
        """
        功能：向屏幕和文件打印输出内容
        :param content: 打印内容
        :return:
        """
        if self.isPrintLog:
            print(content)
        if self.isEvaluate:
            f = open(self.outputAddress + self.evaluateFile, "a")   # 在文件末尾追加
            f.write(content)
            f.write("\n")
            f.close()

    def getROIRegionForIncreMethod(self, image, direction=1, order="first", searchRatio=0.1):
        """
        功能：对于搜索增长方法，根据比例获得其搜索区域
        :param image: 原始图像
        :param direction: 搜索方向
        :param order: 'first' or 'second' 判断属于第几张图像
        :param searchRatio: 裁剪搜素区域的比例，默认搜索方向上的长度的0.1
        :return: 搜索区域
        """
        row, col = image.shape[:2]
        if direction in [1, 3]:  # Vertical directions
            searchLength = int(np.floor(row * searchRatio))
            if (direction == 1 and order == "first") or (direction == 3 and order == "second"):
                roiRegion = image[row - searchLength:row, :]
            else:
                roiRegion = image[0:searchLength, :]
        else:  # Horizontal directions
            searchLength = int(np.floor(col * searchRatio))
            if (direction == 2 and order == "first") or (direction == 4 and order == "second"):
                roiRegion = image[:, col - searchLength:col]
            else:
                roiRegion = image[:, 0:searchLength]
        return roiRegion

    def getROIRegion(self, image, direction="horizontal", order="first", searchLength=150, searchLengthForLarge=-1):
        '''
        功能：对于搜索增长方法，根据固定长度获得其搜索区域（已弃用）
        :param originalImage: 需要裁剪的原始图像
        :param direction: 拼接的方向
        :param order: 该图片的顺序，是属于第一还是第二张图像
        :param searchLength: 搜索区域大小，单位为像素
        :param searchLengthForLarge: 对于行拼接和列拼接的搜索区域大小
        :return: 返回感兴趣区域图像
        '''
        row, col = image.shape[:2]
        start_col, end_col, start_row, end_row = 0, col, 0, row

        if direction in ["horizontal", 2]:
            if searchLengthForLarge != -1:
                searchLength = searchLengthForLarge
            end_col = searchLength if order == "second" else col
            start_col = col - searchLength if order == "first" else 0
        elif direction in ["vertical", 1]:
            if searchLengthForLarge != -1:
                searchLength = searchLengthForLarge
            end_row = searchLength if order == "second" else row
            start_row = row - searchLength if order == "first" else 0

        # Extract the region of interest based on the calculated indices
        roiRegion = image[start_row:end_row, start_col:end_col]
        return roiRegion

    def getOffsetByMode(self, kpsA, kpsB, matches, offsetEvaluate = 10):
        """
        功能：通过求众数的方法获得偏移量
        :param kpsA: 第一张图像的特征
        :param kpsB: 第二张图像的特征
        :param matches: 配准列表
        :param offsetEvaluate: 如果众数的个数大于本阈值，则配准正确，默认为10
        :return: 返回(totalStatus, [dx, dy]), totalStatus 是否正确，[dx, dy]默认[0, 0]
        """
        totalStatus = True
        if len(matches) == 0:
            totalStatus = False
            return (totalStatus, [0, 0])
        dxList = []; dyList = [];
        for trainIdx, queryIdx in matches:
            ptA = (kpsA[queryIdx][1], kpsA[queryIdx][0])
            ptB = (kpsB[trainIdx][1], kpsB[trainIdx][0])
            # dxList.append(int(round(ptA[0] - ptB[0])))
            # dyList.append(int(round(ptA[1] - ptB[1])))
            if int(ptA[0] - ptB[0]) == 0 and int(ptA[1] - ptB[1]) == 0:
                continue
            dxList.append(int(ptA[0] - ptB[0]))
            dyList.append(int(ptA[1] - ptB[1]))
        if len(dxList) == 0:
            dxList.append(0); dyList.append(0)
        # Get Mode offset in [dxList, dyList], thanks for clovermini
        zipped = zip(dxList, dyList)
        zip_list = list(zipped)
        zip_dict = dict((a, zip_list.count(a)) for a in zip_list)
        zip_dict_sorted = dict(sorted(zip_dict.items(), key=lambda x: x[1], reverse=True))

        dx = list(zip_dict_sorted)[0][0]
        dy = list(zip_dict_sorted)[0][1]
        num = zip_dict_sorted[list(zip_dict_sorted)[0]]
        # print("dx = " + str(dx) + ", dy = " + str(dy) + ", num = " + str(num))

        if num < offsetEvaluate:
            totalStatus = False
        # self.printAndWrite("  In Mode, The number of num is " + str(num) + " and the number of offsetEvaluate is "+str(offsetEvaluate))
        return (totalStatus, [dx, dy])

    def getOffsetByRansac(self, kpsA, kpsB, matches, offsetEvaluate=100):
        """
        功能：通过求Ransac的方法获得偏移量（不完善）
        :param kpsA: 第一张图像的特征
        :param kpsB: 第二张图像的特征
        :param matches: 配准列表
        :param offsetEvaluate: 对于Ransac求属于最小范围的个数，大于本阈值，则正确
        :return: 返回(totalStatus, [dx, dy]), totalStatus 是否正确，[dx, dy]默认[0, 0]
        """
        totalStatus = False
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        if len(matches) == 0:
            return (totalStatus, [0, 0], 0)
        # 计算视角变换矩阵
        H1 = cv2.getAffineTransform(ptsA, ptsB)
        # print("H1")
        # print(H1)
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 3, 0.9)
        trueCount = 0
        for i in range(0, len(status)):
            if status[i] == True:
                trueCount = trueCount + 1
        if trueCount >= offsetEvaluate:
            totalStatus = True
            adjustH = H.copy()
            adjustH[0, 2] = 0;adjustH[1, 2] = 0
            adjustH[2, 0] = 0;adjustH[2, 1] = 0
            return (totalStatus ,[np.round(np.array(H).astype(np.int)[1,2]) * (-1), np.round(np.array(H).astype(np.int)[0,2]) * (-1)], adjustH)
        else:
            return (totalStatus, [0, 0], 0)

    def npToListForKeypoints(self, array):
        '''
        功能：Convert array to List, used for keypoints from GPUDLL to python List
        :param array: array from GPUDLL
        :return: kps
        '''
        kps = []
        row, col = array.shape
        for i in range(row):
            kps.append([array[i, 0], array[i, 1]])
        return kps

    def npToListForMatches(self, array):
        '''
        功能：Convert array to List, used for DMatches from GPUDLL to python List
        :param array: array from GPUDLL
        :return: matches
        '''
        descritpors = []
        row, col = array.shape
        for i in range(row):
            descritpors.append((array[i, 0], array[i, 1]))
        return descritpors

    def npToKpsAndDescriptors(self, array):
        """
        功能:？
        :param array:
        :return: kps, descriptors
        """
        kps = []
        descriptors = array[:, :, 1]
        for i in range(array.shape[0]):
            kps.append([array[i, 0, 0], array[i, 1, 0]])
        return (kps, descriptors)

    def detectAndDescribe(self, image, featureMethod):
        '''
    	功能：计算图像的特征点集合，并返回该点集＆描述特征
    	:param image: 需要分析的图像
    	:return: 返回特征点集，及对应的描述特征(kps, features)
    	'''
        if self.isGPUAvailable == False: # CPU mode
            if featureMethod == "sift":
                descriptor = cv2.SIFT_create()
            elif featureMethod == "surf":
                descriptor = cv2.xfeatures2d.SURF_create()
            elif featureMethod == "orb":
                descriptor = cv2.ORB_create(self.orbNfeatures, self.orbScaleFactor, self.orbNlevels, self.orbEdgeThreshold, self.orbFirstLevel, self.orbWTA_K, 0, self.orbPatchSize, self.orbFastThreshold)
            # 检测SIFT特征点，并计算描述子
            kps, features = descriptor.detectAndCompute(image, None)
            # 将结果转换成NumPy数组
            kps = np.float32([kp.pt for kp in kps])
        else:  # GPU mode
            if featureMethod == "sift":
                # 目前GPU-SIFT尚未开发，先采用CPU版本的替代
                descriptor = cv2.SIFT_create()
                kps, features = descriptor.detectAndCompute(image, None)
                kps = np.float32([kp.pt for kp in kps])
            elif featureMethod == "surf":
                # Check if CUDA is available
                if cuda.getCudaEnabledDeviceCount() > 0:
                    # Initialize CUDA SURF detector
                    descriptor = cuda.SURF_CUDA_create(_hessianThreshold=self.surfHessianThreshold, _nOctaves=self.surfNOctaves, _nOctaveLayers=self.surfNOctaveLayers, _extended=self.surfIsExtended, _upright=self.surfIsUpright)
                    
                    # Upload image to GPU
                    gpuImage = cuda.GpuMat()
                    gpuImage.upload(image)
                    
                    # Detect keypoints and compute descriptors with CUDA SURF
                    gpu_kps, gpu_des = descriptor.detectWithDescriptors(gpuImage, None, useProvidedKeypoints=False)
                    
                    # Convert keypoints to format compatible with non-CUDA processing
                    kps = descriptor.downloadKeypoints(gpu_kps)
                    features = gpu_des.download()
                else:
                    print("CUDA not available. Using CPU version of SURF.")
                    # Using OpenCV's built-in SURF if available
                    descriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=self.surfHessianThreshold, nOctaves=self.surfNOctaves, nOctaveLayers=self.surfNOctaveLayers, extended=self.surfIsExtended, upright=self.surfIsUpright)
                    kps, features = descriptor.detectAndCompute(image, None)
                    kps = np.float32([kp.pt for kp in kps])
            elif featureMethod == "orb":
                # Ensure CUDA device is available
                if cuda.getCudaEnabledDeviceCount() > 0:
                    # Initialize CUDA ORB detector
                    descriptor = cuda.ORB_create(nfeatures=self.orbNfeatures, scaleFactor=self.orbScaleFactor, nlevels=self.orbNlevels, edgeThreshold=self.orbEdgeThreshold, firstLevel=self.orbFirstLevel, WTA_K=self.orbWTA_K, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=self.orbPatchSize, fastThreshold=self.orbFastThreshold)
                    
                    # Upload image to GPU
                    gpuImage = cuda.GpuMat()
                    gpuImage.upload(image)
                    
                    # Detect keypoints and compute descriptors with CUDA ORB
                    gpu_kps, gpu_features = descriptor.detectAndCompute(gpuImage, None)
                    
                    # Download keypoints and descriptors to host memory
                    kps = [cv2.KeyPoint(x=kp.pt[0], y=kp.pt[1], _size=kp.size, _angle=kp.angle, _response=kp.response, _octave=kp.octave, _class_id=kp.class_id) for kp in gpu_kps]
                    features = gpu_features.download()
                    kps = np.float32([kp.pt for kp in kps])
                else:
                    print("CUDA device not available. Using CPU-based feature detection.")
                    # Fallback to CPU-based detection (e.g., ORB, SIFT, or original SURF as per your requirement)
                    descriptor = cv2.ORB_create()  # Example fallback to ORB on CPU
                    kps, features = descriptor.detectAndCompute(image, None)
                    kps = np.float32([kp.pt for kp in kps])
        # 返回特征点集，及对应的描述特征
        return (kps, features)

    def matchDescriptors(self, featuresA, featuresB):
        '''
        功能：匹配特征点
        :param featuresA: 第一张图像的特征点描述符
        :param featuresB: 第二张图像的特征点描述符
        :return: 返回匹配的对数matches
        '''
        if self.isGPUAvailable == False:  # CPU Mode
            # 建立暴力匹配器
            if self.featureMethod == "surf" or self.featureMethod == "sift":
                matcher = cv2.DescriptorMatcher_create("BruteForce")
                # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2，返回一个列表
                rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
                matches = []
                for m in rawMatches:
                # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
                    if len(m) == 2 and m[0].distance < m[1].distance * self.searchRatio:
                        # 存储两个点在featuresA, featuresB中的索引值
                        matches.append((m[0].trainIdx, m[0].queryIdx))
            elif self.featureMethod == "orb":
                matcher = cv2.DescriptorMatcher_create("BruteForce-Hamming")
                rawMatches = matcher.match(featuresA, featuresB)
                matches = []
                for m in rawMatches:
                    matches.append((m.trainIdx, m.queryIdx))
            # self.printAndWrite("  The number of matches is " + str(len(matches)))
        else:  # GPU Mode
            if self.featureMethod == "surf":
                # Initialize FLANN based matcher for SURF
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)  # or pass empty dictionary
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                
                matches = flann.knnMatch(np.asarray(featuresA, np.float32), np.asarray(featuresB, np.float32), k=2)
                # Filter matches using the Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < self.searchRatio * n.distance:
                        good_matches.append(m)
                matches = good_matches

            elif self.featureMethod == "orb":
                # Initialize BFMatcher for ORB with default params
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                
                matches = bf.knnMatch(np.asarray(featuresA, np.uint8), np.asarray(featuresB, np.uint8), k=2)
                # Filter matches to satisfy the max distance condition
                good_matches = [m[0] for m in matches if len(m) == 2 and m[0].distance < self.orbMaxDistance]
                matches = good_matches
        return matches

    def resizeImg(self, image, resizeTimes, interMethod = cv2.INTER_AREA):
        """
        功能：缩放图像
        :param image: 原图像
        :param resizeTimes: 缩放比例
        :param interMethod: 插值方法，默认cv2.INTER_AREA
        :return:
        """
        (h, w) = image.shape
        resizeH = int(h * resizeTimes)
        resizeW = int(w * resizeTimes)
        # cv2.INTER_AREA是测试后最好的方法
        return cv2.resize(image, (resizeW, resizeH), interpolation=interMethod)

    def rectifyFinalImg(self, image, regionLength = 10):
        """
        功能：测试用，尚不完善
        :param image: 图像
        :param regionLength: 区域长度
        :return: 返回处理后的图像
        """
        (h, w) = image.shape
        print("h:" + str(h))
        print("w:" + str(w))
        upperLeft   = np.sum(image[0: regionLength, 0: regionLength])
        upperRight  = np.sum(image[0: regionLength, w - regionLength: w])
        bottomLeft  = np.sum(image[h - regionLength: h, 0: regionLength])
        bottomRight = np.sum(image[h - regionLength: h, w - regionLength: w])

        # 预处理
        zeroCol = image[:, 0]
        noneZeroNum = np.count_nonzero(zeroCol)
        zeroNum = h - noneZeroNum
        print("noneZeroNum:" + str(noneZeroNum))
        print("zeroNum:" + str(zeroNum))
        print("division:" + str(noneZeroNum / h))

        # Determine if rotation is needed based on non-zero pixel ratios and corner values
        if (noneZeroNum / h) < 0.3:
            resultImage = image
        else:
            # 左边低，右边高 或 左边高，右边低
            if (upperLeft == 0 and bottomRight == 0 and upperRight != 0 and bottomLeft != 0) or \
            (upperLeft != 0 and bottomRight != 0 and upperRight == 0 and bottomLeft == 0):
                print("Condition met for rotation")
                center = (w // 2, h // 2)
                angle = math.atan(center[1] / center[0]) * 180 / math.pi
                # Determine the direction of rotation based on corner values
                angle = -angle if upperLeft == 0 else angle
                print(f"Center: {center}, Angle: {angle}")
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                resultImage = cv2.warpAffine(image, M, (w, h))
            else:
                resultImage = image
        return resultImage