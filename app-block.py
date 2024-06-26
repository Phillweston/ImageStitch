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
        offsetCalculationMethod,
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
    Stitcher.directIncre = directIncre                   # incremental step for direction adjustment

    stitcher = Stitcher()

    outputAddress = outputFolderPrefix + str.capitalize(Stitcher.fuseMethod) + "\\"
    if enableIncremental:
        if offsetCalculationMethod == 'FeatureSearch':
            return stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, fileNum, stitcher.calculateOffsetForFeatureSearchIncre,
                                startNum=startNum, fileExtension=inputFileExtension, outputfileExtension=outputfileExtension)
        elif offsetCalculationMethod == 'PhaseCorrelate':
            return stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, fileNum, stitcher.calculateOffsetForPhaseCorrelateIncre,
                                startNum=startNum, fileExtension=inputFileExtension, outputfileExtension=outputfileExtension)
        else:
            raise ValueError("Invalid offset calculation method")
    else:
        if offsetCalculationMethod == 'FeatureSearch':
            return stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, fileNum, stitcher.calculateOffsetForFeatureSearch,
                                startNum=startNum, fileExtension=inputFileExtension, outputfileExtension=outputfileExtension)
        elif offsetCalculationMethod == 'PhaseCorrelate':
            return stitcher.imageSetStitchWithMultiple(projectAddress, outputAddress, fileNum, stitcher.calculateOffsetForPhaseCorrelate,
                                startNum=startNum, fileExtension=inputFileExtension, outputfileExtension=outputfileExtension)
        else:
            raise ValueError("Invalid offset calculation method")

with gr.Blocks() as iface:
    gr.Markdown("# Image Stitching Interface \nDesigned by [Phill Weston](https://github.com/Phillweston), all rights reserved.")
    with gr.Tab("Image Stitching Configuration"):
        gr.Markdown("## Image Stitching Configuration\nConfigure your image stitching parameters below.")

        with gr.Column():
            gr.Markdown("### Feature and Performance Settings")
            with gr.Row():
                featureMethod = gr.Radio(choices=["sift", "surf", "orb"], value="sift", label="Feature Method", info="Feature method for image stitching.")
                isColorMode = gr.Checkbox(value=True, label="Color Mode", info="Enable color mode for stitching.")
                windowing = gr.Checkbox(value=True, label="Windowing", info="Enable windowing for better stitching results.")
                isGPUAvailable = gr.Checkbox(value=False, label="GPU Available", info="Utilize GPU for faster processing.")
                isEnhance = gr.Checkbox(value=False, label="Enhance", info="Apply image enhancement before stitching.")
                isClahe = gr.Checkbox(value=False, label="CLAHE", info="Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).")

            gr.Markdown("### Matching and Stitching Settings")
            with gr.Row():
                searchRatio = gr.Slider(minimum=0.1, maximum=1.0, value=0.75, step=0.05, label="Search Ratio", info="Search ratio for feature matching.")
                offsetCalculate = gr.Radio(choices=["mode", "ransac"], value="mode", label="Offset Calculation Method", info="Method to calculate offset between images.")
                offsetEvaluate = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Offset Evaluation Metric", info="Metric to evaluate the calculated offset.")
                roiRatio = gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05, label="ROI Ratio", info="Region of Interest ratio for feature extraction.")
                fuseMethod = gr.Radio(choices=["notFuse", "average", "maximum", "minimum", "fadeInAndFadeOut", "trigonometric", "multiBandBlending", "optimalSeamLine"], value="fadeInAndFadeOut", label="Fuse Method", info="Method to fuse overlapping areas.")

            gr.Markdown("### Direction and Offset Calculation Settings")
            with gr.Row():
                direction = gr.Number(label="Direction", value=1, info="Direction for stitching processing.")
                directIncre = gr.Number(label="Direction Increment", value=0, info="Incremental step for direction adjustment.")
                enableIncremental = gr.Checkbox(value=True, label="Enable Incremental", info="Enable incremental processing for feature searching.")
                offsetCalculationMethod = gr.Radio(choices=["FeatureSearch", "PhaseCorrelate"], value="FeatureSearch", label="Offset Calculation Method", info="Offset calculation method for stitching.")

            gr.Markdown("### Project and Output Settings")
            with gr.Row():
                projectAddress = gr.Textbox(value="demoImages\\iron", label="Project Address", placeholder="Input project address here", info="Directory containing images to stitch.")
                outputFolder = gr.Textbox(value="result\\iron", label="Output Folder Prefix", placeholder="Input output folder prefix here", info="Directory prefix to save stitched results.")
                fileNum = gr.Number(value=1, label="File Number", info="Number of files to process.")
                startNum = gr.Number(value=1, label="Start Number", info="Starting file index.")
                inputFileExtension = gr.Radio(choices=["jpg", "png"], value="jpg", label="Input File Extension", info="File extension for input images.")
                outputfileExtension = gr.Radio(choices=["jpg", "png"], value="jpg", label="Output File Extension", info="File extension for output images.")

        submit_button = gr.Button("Run Stitching")
        submit_button.click(
            stitchWithFeature,
            inputs=[
                featureMethod, isColorMode, windowing, isGPUAvailable, isEnhance, isClahe, searchRatio,
                offsetCalculate, offsetEvaluate, roiRatio, fuseMethod, direction, directIncre, enableIncremental, offsetCalculationMethod,
                projectAddress, outputFolder, fileNum, startNum, inputFileExtension, outputfileExtension
            ],
            outputs=[
                gr.Gallery(label='Saved Images')
            ]
        )

if __name__ == '__main__':
    iface.launch()