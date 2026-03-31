---
language:
- en
dataset_name: AMOD-V1.0
tags:
- Object-Detection
- Synthetic
- Military
- Equipment
- War
- Arma-3
license: cc-by-nc-4.0
pretty_name: Arma 3 Military Object Detection (v1.0 - For Optical Aerial Imagery)
task_categories:
- object-detection
size_categories:
- 10K<n<100K
---


<div align="center">
    <img alt="AMOD: Arma3 Military Object Detection" src="https://raw.githubusercontent.com/unique-chan/AMOD/f5b45bff3b65de89bb72e28b356e8ad665b6e42f/mmrotate/Logo.svg" />
</div>

<hr>

<h3 align="center">
 Dataset for <b>AMOD-V1.0</b>
</h3>

<p align="center">
    <a href="#"><img alt="ARMA3" src="https://img.shields.io/badge/Game-ARMA3-green?logo=steam"></a>
</p>

<hr>

<p align="center">
  Correspondence to
  <a href="mailto:yechankim@gm.gist.ac.kr"><b>Yechan Kim</b></a>,
  <a href="mailto:citizen135@gm.gist.ac.kr"><b>JongHyun Park</b></a>, and 
  <a href="mailto:bluesooyeon@gm.gist.ac.kr"><b>SooYeon Kim</b></a>
</p>

### What is AMOD?
* Here, `AMOD` refers to our large-scale synthetic dataset, <u>A</u>rma3 <u>M</u>ilitary <u>O</u>bject <u>D</u>etection!
* Our **AMOD-V1.0** involves various images and corresponding labels (including oriented bounding boxes) for military object detection in aerial imagery. Each region is captured from six different viewpoints (Look angles: 0°, 10°, 20°, 30°, 40°, 50°).
* For additional information, we direct readers to [our official project homepage](https://sites.google.com/view/yechankim/amod).
  
### How to play with AMOD?
* We highly recommend that users access our [experiment kit](https://github.com/unique-chan/AMOD).
* You can use our [data viewer](https://github.com/unique-chan/AMOD-viewer) to visually explore the images and labels.

### Data structure of AMOD-V1.0
~~~
|—— 📁 .
	|—— 📁 train
		|—— 📁 0000
			|—— 📁 0
                |—— 🖼️ EO_0000_0.jpg
                |—— 📃 ANNOTATION-EO_0000_0.csv
			|—— 📁 10
			|—— ...
			|—— 📁 50
		|—— 📁 ... 
	|—— 📁 test
		|—— 📁 5206
			|—— 📁 0
                |—— 🖼️ EO_5206_0.jpg
            |—— ...
		|—— 📁 ... 
	|—— train.txt
	|—— val.txt
	|—— test.txt
~~~