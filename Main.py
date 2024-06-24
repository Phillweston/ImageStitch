from Stitcher import Stitcher


# Example 1:
# direction: 1, directIncre: 0
# projectAddress: "demoImages\\iron"
# outputFolder: "result\\iron"
# enableIncremental: True

# Example 2:
# direction: 1, directIncre: 1
# projectAddress: "demoImages\\dendriticCrystal"
# outputFolder: "result\\dendriticCrystal"
# enableIncremental: True

# Example 3:
# direction: 4, directIncre: 0
# projectAddress = "demoImages\\zirconBSE"
# outputFolder = "result\\zirconBSE"
# enableIncremental: False

# Example 4:
# direction: 4, directIncre: 0
# projectAddress = "demoImages\\zirconCL"
# outputFolder = "result\\zirconCL"
# enableIncremental: False

# Example 5:
# direction: 4, directIncre: 0
# projectAddress: "demoImages\\zirconREM"
# outputAddress: "result\\zirconREM"
# enableIncremental: False

# Example 6:
# direction: 4, directIncre: 0
# projectAddress: "demoImages\\zirconTEM"
# outputAddress: "result\\zirconTEM"
# enableIncremental: False

def stitchWithFeature():
    Stitcher.featureMethod = "sift"             # "sift","surf" or "orb"
    Stitcher.isColorMode = True                 # True:color, False: gray
    Stitcher.windowing = True                   # Enable windowing for stitching
    Stitcher.isGPUAvailable = False
    Stitcher.isEnhance = False
    Stitcher.isClahe = False
    Stitcher.searchRatio = 0.75                 # 0.75 is common value for matches
    Stitcher.offsetCalculate = "mode"           # "mode" or "ransac"
    Stitcher.offsetEvaluate = 3                 # 3 menas nums of matches for mode, 3.0 menas  of matches for ransac
    Stitcher.roiRatio = 0.2                     # roi length for stitching in first direction
    Stitcher.fuseMethod = "fadeInAndFadeOut"    # "notFuse","average","maximum","minimum","fadeInAndFadeOut","trigonometric", "multiBandBlending"
    stitcher = Stitcher()

    # TODO:
    Stitcher.direction = 1;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\iron"
    outputAddress = "result\\iron" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearchIncre,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 1;  Stitcher.directIncre = 1;
    projectAddress = "demoImages\\dendriticCrystal"
    outputAddress = "result\\dendriticCrystal" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearchIncre,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 4;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\zirconBSE"
    outputAddress = "result\\zirconBSE" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearch,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 4;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\zirconCL"
    outputAddress = "result\\zirconCL" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearch,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 4;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\zirconREM"
    outputAddress = "result\\zirconREM" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearch,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

    Stitcher.direction = 4;  Stitcher.directIncre = 0;
    projectAddress = "demoImages\\zirconTEM"
    outputAddress = "result\\zirconTEM" + str.capitalize(Stitcher.fuseMethod) + "\\"
    stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, 1, stitcher.calculateOffsetForFeatureSearch,
                            startNum=1, fileExtension="jpg", outputfileExtension="jpg")

if __name__=="__main__":
    stitchWithFeature()
