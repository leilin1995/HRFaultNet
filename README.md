# HRFaultNet
High-resolution automatic fault identification method. This is a repository for the paper "HRFaultNet: High-resolution Fault Recognition Using Deep Learning" (submit to TGRS).

## Workflow
An end-to-end workflow for automatic high-resolution fault identification. The original seismic image (a) is fed to the well-trained HRNet to acquire the high-resolution seismic image (b). Then, the well-trained FaultNet obtains the high-resolution fault identification results (c) from the enhanced seismic image.
![image](https://github.com/leilin1995/HRFaultNet/blob/master/workflow.png)

## Example
Fig. a and b display the original seismic data and the corresponding FaultNet predicted fault probability map. Fig. c presents the high-resolution random noise-suppressed seismic image acquired by HRNet from Fig. a. Fig. d shows the FaultNet predicted fault image from Fig. c. Our workflow provides cleaner and sharper fault probability maps than feeding raw seismic data directly into FaultNet. In addition, there is less noise near the fault lines predicted by our method.
![image](https://github.com/leilin1995/HRFaultNet/blob/master/Application/Real/F3/compare.png)

## Project Organization


## Code
All training and test code are in the directory **FaultSegmentation/code** and **ImproveResolution/code**. And the code for field data application and plotting is in the in the directory **Application/Real**.

## Dataset
The synthetic seismic data used for training can be obtained by visting the "".



## Dependencies

* python 3.6.13
* pytorch 1.9.1
* torchvision 0.10.1
* tqdm 4.62.3
* scipy 1.5.4
* numpy 1.19.5
* h5py 3.1.0
* pandas 1.1.5
* PIL 8.4.0
* matplotlib 3.3.4

## Usage instructions
You can use this method following the example in the application.

## Citation

If you find this work useful in your research, please consider citing:

```

```

BibTex

```html

```
