# A Fast Algorithm for Material Image Sequential Stitching

In material research, observing whole microscopic sections at high resolution is often highly desirable. Micrograph stitching, an important technology, produces panoramas or larger images by combining multiple images with overlapping areas while retaining microscopic resolution. However, due to the high complexity and variety of microstructures, most traditional methods struggle to balance the speed and accuracy of the stitching strategy. To overcome this problem, we developed a very fast sequential micrograph stitching method, called VFSMS, which employs an incremental searching strategy and GPU acceleration to guarantee the accuracy and speed of the stitching results. Experimental results demonstrate that VFSMS achieves state-of-the-art performance on six types of microscopic datasets in both accuracy and speed aspects. Besides, it significantly outperforms the most famous and commonly used software, such as ImageJ, Photoshop, and Autostitch.

## Note

This repository contains unstable GPU acceleration and will not provide support from now on. A new version of the software, equipped with stable GPU acceleration, multi-process acceleration and more advanced stitching algorithm, can be downloaded and installed at [the website](http://microstitch.tech/).

## Requirements

Python 3 needs to be installed before running this script.
To run this algorithm, you need to install the Python packages as follows:

    opencv-contrib (we have tested OpenCV 4.10 and Python 3.12)

As we have tested, python 3.7 could only support OpenCV 4.0 which has totally no sift or cuda-sift in the contrib package. We recommend using Python 3.6.

## (Deprecated) GPU Version in Windows

We rebuilt the opencv-contrib 3.3.1 and cuda9.0 in our code and provided these DLL files. If you want to use it, please unrar it in the project address.

### Be careful

Surf cuda in Opencv is good at feature search and not good at feature match. It will raise an error if the graphic memory is insufficient.

## Examples

There are some examples of VFSMS are shown behind.

Six types of local and global micrographs and their shooting paths. The red translucent region represents one shot from the microscope. The red dotted line refers to the shooting path. (a) Iron crystal in scanning electron microscopy (SEM) with its detailed imaging. (b) Pairwise shooting path of (a) with 2 local images. (c) Dendritic crystal in SEM with its detailed imaging. (d) Grid shooting path of (c) with 90 local images. (e) Zircon in SEM with its detailed imaging. (f) Zircon in transmission electron microscopy (TEM) with its detailed imaging. (g) Zircon in backscattered electron imaging (BSE) with its detailed imaging. (h) Zircon in cathodoluminescence spectroscopy (CL) with its detailed imaging. (i) Shooting path for (e), (f), (g), (h); the number of local images depends on the length of the sample.

<p align = "center">
    <img src="./demoImages/examplesOfImageStitch.png">
</p>

## Citation

If you use it successfully for your research please be so kind as to cite [the paper](https://www.sciencedirect.com/science/article/pii/S0927025618307158).

Ma B, Ban X, Huang H, et al. A fast algorithm for material image sequential stitching [J]. Computational Materials Science, 2019, 158: 1-13.

or

    @article{MA20191,
    title = "A fast algorithm for material image sequential stitching",
    journal = "Computational Materials Science",
    volume = "158",
    pages = "1 - 13",
    year = "2019",
    issn = "0927-0256",
    doi = "https://doi.org/10.1016/j.commatsci.2018.10.044",
    url = "http://www.sciencedirect.com/science/article/pii/S0927025618307158",
    author = "Boyuan Ma and Xiaojuan Ban and Haiyou Huang and Wanbo Liu and Chuni Liu and Di Wu and Yonghong Zhi"}
