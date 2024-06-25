import numpy as np
import cv2
import ImageUtility as Utility


class ImageFusion(Utility.Method):

    isColorMode = False

    # 图像融合类，目前只编写传统方法
    def fuseByAverage(self, images):
        '''
        功能：均值融合
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        '''
        (imageA, imageB) = images
        # 由于相加后数值可能溢出，需要转变类型
        fuseRegion = np.uint8((imageA.astype(int) + imageB.astype(int)) / 2)
        return fuseRegion

    def fuseByMaximum(self, images):
        '''
        功能：最大值融合
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        '''
        (imageA, imageB) = images
        fuseRegion = np.maximum(imageA, imageB)
        return fuseRegion

    def fuseByMinimum(self, images):
        '''
        功能：最小值融合
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        '''
        (imageA, imageB) = images
        fuseRegion = np.minimum(imageA, imageB)
        return fuseRegion

    def getWeightsMatrix(self, images):
        '''
        功能：获取权值矩阵
        :param images:  输入两个相同区域的图像
        :return: 两个权值矩阵
        '''
        (imageA, imageB) = images
        row, col = imageA.shape[:2]
        weightMatA = np.ones(imageA.shape, dtype=np.float32)
        weightMatB = np.ones(imageA.shape, dtype=np.float32)
        
        # Efficient comparison using numpy
        compareList = [np.count_nonzero(imageA[:row // 2, :col // 2] > 0),
                    np.count_nonzero(imageA[row // 2:, :col // 2] > 0),
                    np.count_nonzero(imageA[row // 2:, col // 2:] > 0),
                    np.count_nonzero(imageA[:row // 2, col // 2:] > 0)]

        index = np.argmin(compareList)

        # Use numpy broadcasting for weight assignments
        if index == 2:  # Top-left quadrant
            rowIndex = np.argmax(np.any(imageA[:, :-1] != -1, axis=0))
            colIndex = np.argmax(np.any(imageA[:-1, :] != -1, axis=1))
            weightMatB[:rowIndex + 1, :] *= np.linspace(0, 1, rowIndex + 1)[:, None]
            weightMatB[:, :colIndex + 1] *= np.linspace(0, 1, colIndex + 1)

        elif index == 3:  # Bottom-left quadrant
            rowIndex = np.argmax(np.any(imageA[:, :-1] != -1, axis=0))
            colIndex = np.argmax(np.any(imageA[1:, :] != -1, axis=1))
            weightMatB[rowIndex:, :] *= np.linspace(1, 0, row - rowIndex)[:, None]
            weightMatB[:, :colIndex + 1] *= np.linspace(0, 1, colIndex + 1)

        elif index == 0:  # Bottom-right quadrant
            rowIndex = np.argmax(np.any(imageA[:, 1:] != -1, axis=0))
            colIndex = np.argmax(np.any(imageA[1:, :] != -1, axis=1))
            weightMatB[rowIndex:, :] *= np.linspace(1, 0, row - rowIndex)[:, None]
            weightMatB[:, colIndex:] *= np.linspace(1, 0, col - colIndex)

        elif index == 1:  # Top-right quadrant
            rowIndex = np.argmax(np.any(imageA[:, 1:] != -1, axis=0))
            colIndex = np.argmax(np.any(imageA[:-1, :] != -1, axis=1))
            weightMatB[:rowIndex + 1, :] *= np.linspace(0, 1, rowIndex + 1)[:, None]
            weightMatB[:, colIndex:] *= np.linspace(1, 0, col - colIndex)

        weightMatA = 1 - weightMatB
        return weightMatA, weightMatB

    def fuseByFadeInAndFadeOut(self, images, dx, dy):
        '''
        功能：渐入渐出融合
        :param images:输入两个相同区域的图像
        :param dx: x方向偏移
        :param dy: y方向偏移
        :return: 融合后的图像
        '''
        (imageA, imageB) = images
        row, col = imageA.shape[:2]
        weightMatA = np.ones_like(imageA, dtype=np.float32)
        weightMatB = np.ones_like(imageB, dtype=np.float32)

        if np.count_nonzero(imageA > -1) / imageA.size > 0.65:
            # Straight blending
            axis, size = (1, col) if col <= row else (0, row)
            gradient = np.linspace(0, 1, size, endpoint=False, dtype=np.float32)
            if (dy >= 0 and axis == 1) or (dx <= 0 and axis == 0):
                weightMatA = np.apply_along_axis(lambda x: gradient[::-1], axis, weightMatA)
                weightMatB = np.apply_along_axis(lambda x: gradient, axis, weightMatB)
            else:
                weightMatA = np.apply_along_axis(lambda x: gradient, axis, weightMatA)
                weightMatB = np.apply_along_axis(lambda x: gradient[::-1], axis, weightMatB)
        else:
            # 如果对于imageA中，非0值占比例比较小，则认为是拐角融合
            # self.printAndWrite("拐角融合")
            weightMatA, weightMatB = self.getWeightsMatrix(images)

        # Resolve negative pixels
        imageA[imageA < 0] = imageB[imageA < 0]

        # Weighted sum of images
        result = weightMatA * imageA.astype(np.int32) + weightMatB * imageB.astype(np.int32)
        np.clip(result, 0, 255, out=result)

        return result.astype(np.uint8)

    def fuseByTrigonometric(self, images, dx, dy):
        '''
        功能：三角函数融合
        引用自《一种三角函数权重的图像拼接算法》知网
        :param images: 输入两个相同区域的图像
        :param dx: x方向偏移
        :param dy: y方向偏移
        :return: 融合后的图像
        '''
        (imageA, imageB) = images
        row, col = imageA.shape[:2]
        weightMatA = np.ones(imageA.shape, dtype=np.float64)
        weightMatB = np.ones(imageA.shape, dtype=np.float64)
        # self.printAndWrite("    ratio: " + str(np.count_nonzero(imageA > -1) / imageA.size))
        if np.count_nonzero(imageA > -1) / imageA.size > 0.65:
            # 如果对于imageA中，非0值占比例比较大，则认为是普通融合
            # 根据区域的行列大小来判断，如果行数大于列数，是水平方向
            if col <= row:
                # self.printAndWrite("普通融合-水平方向")
                indices = np.arange(col) if dy >= 0 else np.flip(np.arange(col))
                weightMatA *= np.tile(indices * 1.0 / col, (row, 1))
                weightMatB *= np.flip(weightMatA, axis=1)
            # 根据区域的行列大小来判断，如果列数大于行数，是竖直方向
            else:
                indices = np.arange(row) if dx <= 0 else np.flip(np.arange(row))
                weightMatA *= np.tile(indices * 1.0 / row, (col, 1)).T
                weightMatB *= np.flip(weightMatA, axis=0)
        else:
            # 如果对于imageA中，非0值占比例比较小，则认为是拐角融合
            # self.printAndWrite("拐角融合")
            weightMatA, weightMatB = self.getWeightsMatrix(images)

        weightMatA = np.power(np.sin(weightMatA * np.pi / 2), 2)
        weightMatB = 1 - weightMatA

        imageA[imageA < 0] = imageB[imageA < 0]
        result = weightMatA * imageA.astype(np.int) + weightMatB * imageB.astype(np.int)
        result = np.clip(result, 0, 255)
        return np.uint8(result)

    # 多样条融合方法
    def fuseByMultiBandBlending(self, images):
        """
        功能：多带样条融合
        :param images: 输入两个相同区域的图像
        :return: 融合后的图像
        """
        (imageA, imageB) = images
        imagesReturn = np.uint8(self.BlendArbitrary2(imageA, imageB, 4))
        return imagesReturn

    def BlendArbitrary(self, img1, img2, R, level):
        """
        功能：带权拉普拉斯融合
        :param img1: 第一张图像
        :param img2: 第二张图像
        :param R: 权值矩阵
        :param level: 金字塔权重
        :return: 融合后的图像
        """
        # img1 and img2 have the same size
        # R represents the region to be combined
        # level is the expected number of levels in the pyramid

        LA, GA = self.LaplacianPyramid(img1, level)
        LB, GB = self.LaplacianPyramid(img2, level)
        GR = self.GaussianPyramid(R, level)
        GRN = []
        for i in range(level):
            GRN.append(np.ones((GR[i].shape[0], GR[i].shape[1])) - GR[i])
        LC = []
        for i in range(level):
            LC.append(LA[i] * GR[level - i -1] + LB[i] * GRN[level - i - 1])
        result = self.reconstruct(LC)
        return result

    def BlendArbitrary2(self, img1, img2, level):
        """
        Blends two images using an arbitrary blending technique.

        Args:
            img1: The first input image.
            img2: The second input image.
            level: The expected number of levels in the pyramid.

        Returns:
            The blended image.

        Notes:
            - img1 and img2 should have the same size.
            - The blending is performed by combining the Laplacian pyramids of the two images.
            - The resulting pyramid is reconstructed to obtain the final blended image.
        """
        LA, GA = self.LaplacianPyramid(img1, level)
        LB, GB = self.LaplacianPyramid(img2, level)
        LC = []
        for i in range(level):
            LC.append(LA[i] * 0.5 + LB[i] * 0.5)
        result = self.reconstruct(LC)
        return result

    def LaplacianPyramid(self, img, level):
        """
        Constructs a Laplacian pyramid for the given image.

        Args:
            img: The input image.
            level: The number of levels in the pyramid.

        Returns:
            lp: A list of Laplacian images at each level of the pyramid.
            gp: A list of Gaussian images at each level of the pyramid.
        """
        gp = self.GaussianPyramid(img, level)
        lp = [gp[level-1]]
        for i in range(level - 1, -1, -1):
            GE = cv2.pyrUp(gp[i])
            GE = cv2.resize(GE, (gp[i - 1].shape[1], gp[i - 1].shape[0]), interpolation=cv2.INTER_CUBIC)
            L = cv2.subtract(gp[i - 1], GE)
            lp.append(L)
        return lp, gp

    def reconstruct(self, input_pyramid):
        """
        Reconstructs the image from the input pyramid.

        Args:
            input_pyramid (list): A list of images forming the input pyramid.

        Returns:
            numpy.ndarray: The reconstructed image.
        """
        out = input_pyramid[0]
        for i in range(1, len(input_pyramid)):
            out = cv2.pyrUp(out)
            out = cv2.resize(out, (input_pyramid[i].shape[1],input_pyramid[i].shape[0]), interpolation = cv2.INTER_CUBIC)
            out = cv2.add(out, input_pyramid[i])
        return out

    def GaussianPyramid(self, R, level):
        """
        Constructs a Gaussian pyramid from the input image.

        Parameters:
        - R: The input image.
        - level: The number of levels in the pyramid.

        Returns:
        - gp: The Gaussian pyramid as a list of images.

        """
        G = R.copy().astype(np.float64)
        gp = [G]
        for i in range(level):
            G = cv2.pyrDown(G)
            gp.append(G)
        return gp

    #权值矩阵归一化
    def stretchImage(self, Region):
        """
        Stretch the input image region to enhance its contrast.

        Parameters:
        - Region: numpy.ndarray
            The input image region to be stretched.

        Returns:
        - out: numpy.ndarray
            The stretched image region.

        """
        minI = Region.min()
        maxI = Region.max()
        out = (Region - minI) / (maxI - minI) * 255
        return out

    def convertImageDepth(self, image):
        """
        Converts the depth of an image to a desired format.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The image with the converted depth.

        """
        # Check if image depth is CV_16F, convert to CV_32F
        if image.dtype == np.float16:
            image = image.astype(np.float32)
        # Check if image depth is CV_32S, convert to CV_8U
        elif image.dtype == np.int32:
            # Normalize and convert
            image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            image = image.astype(np.uint8)
        return image

    # OptialSeamLine's method 最佳缝合线方法
    def fuseByOptimalSeamLine(self, images, direction="horizontal"):
        '''
        基于最佳缝合线的融合方法
        :param images:输入两个相同区域的图像
        :param direction: 横向拼接还是纵向拼接
        :return: 融合后的图像
        '''
        (imageA, imageB) = images
        # cv2.imshow("imageA", imageA)
        # cv2.imshow("imageB", imageB)
        # cv2.waitKey(0)
        value = self.calculateValue(images)
        # print(value)
        mask = 1 - self.findOptimalSeamLine(value, direction)
        # cv2.namedWindow("mask", 0)
        # cv2.imshow("mask", (mask*255).astype(np.uint8))
        # cv2.waitKey(0)
        fuseRegion = imageA.copy()
        fuseRegion[(1 - mask) == 0] = imageA[(1 - mask) == 0]
        fuseRegion[(1 - mask) == 1] = imageB[(1 - mask) == 1]
        drawFuseRegion = self.drawOptimalLine(1- mask, fuseRegion)
        cv2.imwrite("optimalLine.jpg", drawFuseRegion)
        cv2.imwrite("fuseRegion.jpg", np.uint8(self.BlendArbitrary(imageA,imageB, mask, 4)))
        cv2.waitKey(0)
        return np.uint8(self.BlendArbitrary(imageA,imageB, mask, 4))

    def calculateValue(self, images):
        """
        Calculates the value for image fusion based on color and geometry differences.

        Args:
            images (tuple): A tuple containing two images (imageA and imageB).

        Returns:
            numpy.ndarray: The calculated value for image fusion.

        """
        (imageA, imageB) = images
        row, col = imageA.shape[:2]
        # value = np.zeros(imageA.shape, dtype=np.float32)
        Ecolor = (imageA - imageB).astype(np.float32)
        Sx = np.array([[-2, 0, 2],
                       [-1, 0, 1],
                       [-2, 0, 2]])
        Sy = np.array([[-2, -1, -2],
                       [ 0,  0,  0],
                       [ 2,  1,  2]])
        Egeometry = np.power(cv2.filter2D(Ecolor, -1, Sx), 2) + np.power(cv2.filter2D(Ecolor, -1, Sy), 2)

        diff = np.abs(imageA - imageB) / np.maximum(imageA, imageB).max()
        diffMax = np.amax(diff)

        infinet = 10000
        W = 10
        for i in range(0, row):
            for j in range(0, col):
                if diff[i, j] < 0.7 * diffMax:
                    diff[i, j] = W * diff[i, j] / diffMax
                else:
                    diff[i, j] = infinet
        value = diff * (np.power(Ecolor, 2) + Egeometry)
        return value

    def findOptimalSeamLine(self, value, direction="horizontal"):
        """
        功能：寻找最佳缝合线
        :param value: 计算的值
        :param direction: 方向
        :return: 最佳缝合线
        """
        if direction == "vertical":
            value = np.transpose(value)
        row, col = value.shape[:2]
        dpMatrix = np.zeros_like(value, dtype=np.float32)
        indexMatrix = np.zeros_like(value, dtype=np.int)

        dpMatrix[0, :] = value[0, :]
        indexMatrix[0, :] = -1

        for i in range(1, row):
            for j in range(col):
                left = dpMatrix[i - 1, j - 1] if j > 0 else np.inf
                middle = dpMatrix[i - 1, j]
                right = dpMatrix[i - 1, j + 1] if j < col - 1 else np.inf
                dpMatrix[i, j] = min(left, middle, right) + value[i, j]
                indexMatrix[i, j] = np.argmin([left, middle, right]) - 1 if j > 0 else 0

        # Generate the mask
        mask = np.zeros_like(value, dtype=np.uint8)
        index = dpMatrix[-1].argmin()
        mask[-1, index:] = 1

        for i in range(row - 1, 0, -1):
            index += indexMatrix[i, index]
            mask[i - 1, index:] = 1

        if direction == "vertical":
            mask = np.transpose(mask)

        return mask

    def drawOptimalLine(self, mask, fuseRegion):
        """
        功能：绘制最佳缝合线
        :param mask: 最佳缝合线
        :param fuseRegion: 融合后的图像
        :return: 绘制后的图像
        """
        row, col = mask.shape[:2]
        drawing = np.zeros([row, col, 3], dtype=np.uint8)
        # Before color conversion, convert fuseRegion to a supported depth if necessary
        fuseRegion = cv2.convertScaleAbs(fuseRegion)
        drawing = cv2.cvtColor(fuseRegion, cv2.COLOR_GRAY2BGR)
        for j in range(0, col):
            for i in range(0, row):
                if mask[i, j] == 1:
                    drawing[i, j] = np.array([0, 0, 255])
                    break
        return drawing

if __name__=="__main__":
    # 测试
    num = 6
    A_1 = np.zeros((num, num), dtype=np.uint8)
    for i in range(num):
        for j in range(num):
            if j < 3:
                A_1[i, j] = 1
    for i in range(num):
        for j in range(num):
            if i < 3:
                A_1[i, j] = 1
    # A_1[0, num-1] = 0;A_1[1, num-1] = 0;A_1[2, num-1] = 0;
    # A_1[num-1, 0] = 0;  A_1[num-1, 1] = 0;A_1[num-1, 2] = 0;
    print(A_1)

    A_2 = np.ones((num, num), dtype=np.uint8)
    imageFusion = ImageFusion()
    imageFusion.fuseByFadeInAndFadeOut([A_1, A_2])