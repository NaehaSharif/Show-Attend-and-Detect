# Show-Attend-and-Detect
More than 55,000 people world-wide die from Cardiovascular Disease (CVD) each day. Calcification of the abdominal aorta is an established marker of asymptomatic CVD. It can be observed on scans taken for vertebral fracture assessment from Dual Energy X-ray Absorptiometry machines. Assessment of Abdominal Aortic Calcification (AAC) and timely intervention may help to reinforce public health messages around CVD risk factors and improve disease management, reducing the global health burden related to CVDs. Our research addresses this problem by proposing a novel and reliable framework for automated “fine-grained” assessment of AAC. Inspired by the vision-to-language models, our method performs sequential scoring of calcified lesions along the length of the abdominal aorta on DXA scans; mimicking the human scoring process.

## Network Architecture

![Alt](images/architecture.PNG) 
Our proposed (a) framework for automatic fine-grained AAC scoring,(b) detailed schematic of our attention-based decoders, and (c) attention module.

## Environment setup
Clone this repository and create the `aac-release` conda environment using the `requirements.yml` file:
```
conda env create -f requirements.yml
conda activate aac-release
```

Note: Pytorch 1.7 and Python 3.6 is required to run our code. 


## Pre-processing

Place the dual energy Dicom images in the "dicom_folder" and run the following script to preprocess the images. 

```
preprocessing-DE.py
```
The preprocessed images will be save in the "npy_folder"

Run "preparing-data.py" to read the csv file containing the ids, score and dates of the dicoms and create numpy and csv files
containing the name to score mapping. These mapping files will be used for training and validation purposes.

## Training Script

Run the following script, once for the anterior segment score and once for the posterior segement scores.

```
training.py
```
For training on anterior segments, set the parameter -level='ant', and for training on posterior segments set the parameter -level='post' within the script. 
Once the model files and Csvs (containing fine-grained scores) for anterior segemnts are created after 10-fold cross validation, place them in the folders -> attempt/ant/models, and attempt/ant/fine_scores respectively. Do the same for models trained on posterior segment scores.

## Inference

Run the following script for inference and visualisation.

```
inference_model.py
```

combine anterior and posterior scores by running "combine_ant_and_post.py", and then run "inference_model_combined.py" to generate overall scores. 

## Evaluation

![Alt](images/comparison.PNG) 
Scatter plots and confusion matrix of fine-grained ground truth scores vs predicted scores for our proposed model Mfgs and ground truth vs our implementation of the baseline Mbase [15] overall AAC-24 score per scan.

## Qualitative Results

![Alt](images/visualisation.jpg) 
Figure 4 (refer to our paper)

Our qualitative results show the attention maps generated by our decoding pipeline.
Figure 4 shows some examples where our model succeeds (a-b) or fails (cd). The four sub-figures in each section are from four different time stamps of our sequential attention model. The model “sees” a particular vertebrae at a given time stamp, “attends” to it and “detects” the amount of calcification. It then moves on to the next vertebrae in the sequence. Figure 4(a-b) show how the model attends to each vertebrae and correctly scores the calcification.
Figure 4(c) shows failure cases where the model over-estimates the score of L3 while (d) portrays a case where it totally fails to identify the heavy calcification. However, Figure 4(d) is very interesting as the aorta in the DXA scan produced by the GE iDXA machine is masked for radiation dose reduction. The human experts have not scored L2 and L3 anterior sections of this scan because they are not visible. Our model is unable to “see” the aorta and hence outputs a zero score. (This is good because a higher predicted score would have meant that the model is not paying attention to the aorta in the score generation process).

#### References and Acknowledgements

-https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

-https://github.com/MakarovArtyom/Image-Captioning-with-Attention

-https://medium.com/analytics-vidhya/image-captioning-with-attention-part-1-e8a5f783f6d3

-https://github.com/Bjarten/early-stopping-pytorch/

