# Evaluating Fairness in Deep Learning Approaches for DCE-MRI Breast Cancer Diagnosis

Justin Chang, Sarah Rodwin, Jenny Mao

Professor: Nisheeth K. Vishnoi

##### Table of Contents  
* [Overview](#overview)  
* [Installation](#installation)  
* [Usage](#usage)   
* [Demo(s)](#demos)
* [File Breakdown](#file-breakdown) 
* [Acknowledgements](#acknowledgements)

<a name="overview"/>

# Overview

Dynamic contrast-enhanced (DCE) magnetic resonance imaging (MRI) is a powerful tool in generating radiological images for diagnosing breast cancer. Deep learning holds promise in this area, but the fairness of state-of-the-art diagnosis algorithms have yet to be verified.

<a name="installation"/>

# Installation

The following instructions recapitulate aliases.sh:

```bash
conda install keras
conda install tensorflow
conda install torchvision -c pytorch
conda install tqdm
conda install pandas
conda install -c conda-forge matplotlib
conda install opencv
conda install scikit-learn
pip install pydicom
```

<a name="usage"/>

# Usage

To run the CNN, please see main.py. To run the NYU algorithm, please see train.py. Note that the absolute paths used to read data must be modified. The dataset, Duke-Breast-Cancer-MRI, can be downloaded from https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903. Please use image_data as the target of the NBIA data retriever.

<a name="demos"/>

# Demo(s)

These video(s) may take a few seconds to load!

https://user-images.githubusercontent.com/39058024/196014602-3983c1c4-0622-42a1-979b-399c0cba9b77.mp4

https://user-images.githubusercontent.com/39058024/196014603-aabb0b38-e9df-4482-8379-211c7d9a6704.mp4

The example below shows what augmented images may look like:

https://user-images.githubusercontent.com/39058024/205505004-d3e53fb9-f7ae-45d5-9607-5249b086e8fa.mp4

<a name="file-breakdown"/>

# File Breakdown

<a name="acknowledgements"/>

# Acknowledgements

We thank Dr. Vishnoi for teaching the course "Algorithms and their Societal Implications" and providing mentorship.
