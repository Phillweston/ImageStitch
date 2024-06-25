from Stitcher import Stitcher
import gradio as gr


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
        return stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, fileNum, stitcher.calculateOffsetForFeatureSearchIncre,
                                startNum=startNum, fileExtension=inputFileExtension, outputfileExtension=outputfileExtension)
    else:
        return stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, fileNum, stitcher.calculateOffsetForFeatureSearch,
                                startNum=startNum, fileExtension=inputFileExtension, outputfileExtension=outputfileExtension)

iface = gr.Interface(
    fn=stitchWithFeature,
    inputs=[
        gr.Radio(choices=["sift", "surf", "orb"], value="sift", label="Feature Method", info="Feature method for image stitching."),
        gr.Checkbox(value=True, label="Color Mode", info="Enable color mode for stitching, true for color, false for gray."),
        gr.Checkbox(value=True, label="Windowing", info="Enable windowing for stitching, true for enable, false for disable."),
        gr.Checkbox(value=False, label="GPU Available", info="Enable GPU for stitching, true for GPU, false for CPU."),
        gr.Checkbox(value=False, label="Enhance", info="Enable enhance for stitching, true for enhance, false for not enhance."),
        gr.Checkbox(value=False, label="CLAHE", info="Enable CLAHE for stitching, true for CLAHE, false for not CLAHE."),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Search Ratio", info="Search ratio for matches."),
        gr.Radio(choices=["mode", "ransac"], value="mode", label="Offset Calculate", info="Offset calculate method for stitching."),
        gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Offset Evaluate", info="Offset evaluate for stitching."),
        gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05, label="ROI Ratio", info="ROI ratio for stitching in first direction."),
        gr.Radio(choices=["fadeInAndFadeOut", "notFuse", "average", "maximum", "minimum", "trigonometric", "multiBandBlending", "optimalSeamLine"], value="fadeInAndFadeOut", label="Fuse Method", info="Fuse method for stitching."),
        gr.Slider(minimum=0, maximum=4, step=1, value=1, label="Direction", info="Direction for stitching."),
        gr.Slider(minimum=-1, maximum=1, step=1, value=0, label="Direction Increment", info="Direction increment for stitching."),
        gr.Checkbox(value=True, label="Enable Incremental Method to Calculate Offset for Feature Searching", info="Enable Incremental Method to Calculate Offset for Feature Searching."),
        gr.Textbox(label="Project Address", value="demoImages\\iron", placeholder="Input project address here", info="Input project address for stitching."),
        gr.Textbox(label="Output Folder Prefix", value="result\\iron", placeholder="Input output folder prefix here", info="Input output folder prefix for stitching."),
        gr.Number(label="File Number", value=1, info="Input file number for stitching."),
        gr.Number(label="Start Number", value=1, info="Input start number for stitching."),
        gr.Radio(choices=["jpg", "png"], value="jpg", label="Input File Extension", info="Input file extension for stitching."),
        gr.Radio(choices=["jpg", "png"], value="jpg", label="Output File Extension", info="Output file extension for stitching.")
    ],
    outputs=[
        gr.Gallery(label='Saved Images')
    ],
    title="Image Stitching Interface",
    description="Designed by Phill Weston, all rights reserved."
)

if __name__ == "__main__":
    iface.launch()