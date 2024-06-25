from Stitcher import Stitcher
import argparse


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

def stitchWithFeature(
        featureMethod,
        isColorMode,
        windowing,
        isGPUAvailable,
        isEnhance,
        isClahe,
        searchRatio,
        offsetCalculate,
        offsetEvaluate,
        roiRatio,
        fuseMethod,
        direction,
        directIncre,
        enableIncremental,
        projectAddress,
        outputFolderPrefix,
        fileNum,
        startNum,
        inputFileExtension,
        outputfileExtension
    ):
    Stitcher.featureMethod = featureMethod               # "sift", "surf" or "orb"
    Stitcher.isColorMode = isColorMode                   # True: color, False: gray
    Stitcher.windowing = windowing                       # Enable windowing for stitching
    Stitcher.isGPUAvailable = isGPUAvailable             # True: GPU, False: CPU
    Stitcher.isEnhance = isEnhance                       # True: enhance, False: not enhance
    Stitcher.isClahe = isClahe                           # True: clahe, False: not clahe
    Stitcher.searchRatio = searchRatio                   # 0.75 is common value for matches
    Stitcher.offsetCalculate = offsetCalculate           # "mode" or "ransac"
    Stitcher.offsetEvaluate = offsetEvaluate             # 3 means nums of matches for mode, 3.0 means num of matches for ransac
    Stitcher.roiRatio = roiRatio                         # roi length for stitching in first direction
    Stitcher.fuseMethod = fuseMethod                     # "notFuse", "average", "maximum", "minimum", "fadeInAndFadeOut", "trigonometric", "multiBandBlending" and "optimalSeamLine"
    Stitcher.direction = direction                       # direction for stitching
    Stitcher.directIncre = directIncre                   # direction increment for stitching
    outputAddress = outputFolderPrefix + str.capitalize(Stitcher.fuseMethod) + "\\"
    Stitcher.outputAddress = outputAddress
    stitcher = Stitcher()

    if enableIncremental:
        stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, fileNum, stitcher.calculateOffsetForFeatureSearchIncre,
                                startNum=startNum, fileExtension=inputFileExtension, outputfileExtension=outputfileExtension)
    else:
        stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, fileNum, stitcher.calculateOffsetForFeatureSearch,
                                startNum=startNum, fileExtension=inputFileExtension, outputfileExtension=outputfileExtension)

    return f"Stitching completed! Check the output directory: {outputAddress}"

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Stitch images with feature detection.')
    parser.add_argument('--featureMethod', default='sift', choices=['sift', 'surf', 'orb'], help='Feature detection method')
    parser.add_argument('--isColorMode', type=bool, default=True, help='Color mode (True for color, False for gray)')
    parser.add_argument('--windowing', type=bool, default=True, help='Enable windowing for stitching')
    parser.add_argument('--isGPUAvailable', type=bool, default=False, help='GPU availability')
    parser.add_argument('--isEnhance', type=bool, default=False, help='Enable enhancement')
    parser.add_argument('--isClahe', type=bool, default=False, help='Enable CLAHE')
    parser.add_argument('--searchRatio', type=float, default=0.75, help='Search ratio for matches')
    parser.add_argument('--offsetCalculate', default='mode', choices=['mode', 'ransac'], help='Offset calculation method')
    parser.add_argument('--offsetEvaluate', type=int, default=3, help='Offset evaluation metric')
    parser.add_argument('--roiRatio', type=float, default=0.2, help='ROI ratio for stitching')
    parser.add_argument('--fuseMethod', default='fadeInAndFadeOut', help='Fusion method for stitching')
    parser.add_argument('--direction', type=int, default=1, help='Stitching direction')
    parser.add_argument('--directIncre', type=int, default=0, help='Direction increment')
    parser.add_argument('--enableIncremental', type=bool, default=True, help='Enable incremental stitching')
    parser.add_argument('--projectAddress', default='demoImages\\iron', help='Project images directory')
    parser.add_argument('--outputFolderPrefix', default='result\\iron', help='Output folder prefix')
    parser.add_argument('--fileNum', type=int, default=1, help='Number of files to stitch')
    parser.add_argument('--startNum', type=int, default=1, help='Starting file number')
    parser.add_argument('--inputFileExtension', default='jpg', help='Input file extension')
    parser.add_argument('--outputfileExtension', default='jpg', help='Output file extension')
    args = parser.parse_args()

    stitchWithFeature(
        args.featureMethod,
        args.isColorMode,
        args.windowing,
        args.isGPUAvailable,
        args.isEnhance,
        args.isClahe,
        args.searchRatio,
        args.offsetCalculate,
        args.offsetEvaluate,
        args.roiRatio,
        args.fuseMethod,
        args.direction,
        args.directIncre,
        args.enableIncremental,
        args.projectAddress,
        args.outputFolderPrefix,
        args.fileNum,
        args.startNum,
        args.inputFileExtension,
        args.outputfileExtension
    )
