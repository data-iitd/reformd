# REFORMD
Artifacts related to the CIKM 2021 paper "Region Invariant Normalizing Flows for Mobility Transfer".

## Requirements
Use a python-3.7 environment and install Tensorflow v1.13.1 and Tensorflow-Probability v0.6.0. More details are given in requirements.txt file.

## Execution Instructions
### Dataset Format
We have also provided both the datasets used in the paper [here](https://drive.google.com/drive/folders/1fRoiJNi4TwkDklmLuBYbBWjq-xkWixtO?usp=sharing). Make sure you download them and keep the files inside the 'data' folder before running the code. For any new dataset, you need to structure it to different files containing the POI details and the mobility details of the users as follows:
```
Train_Cat Train_Dist Train_Time Test_Cat Test_Dist Test_Time
```
### File Details
Here, we provide the description of the files that are given in the dataset:
- Train_Cat = The categories of POIs visited by the users in the network.
- Train_Time = The time of the check-in
- Train_Dist = The distance covered by the users during check-ins.

### Running the Code
Use the following command to run REFORMD and provide source and target dataset. For example, to run the model on New York as source and Michigan as target, use the command:
```
python run.py NY_US MI_US
```
Once you run the code, the following procedure steps into action:
- REFORMD loads the source and target datasets and trains the model on source region and then on target region data.
- At the end, it predicts the accuracy and MAE for the target region.

## Citing
If you use this code in your research, please cite:
```
@inproceedings{vinayakcikm,
 author = {Vinayak Gupta and Srikanta Bedathur},
 booktitle = {Proc. of the 30th ACM Intl. Conference on Information and Knowledge Management (CIKM)},
 title = {Region-Invariant Normalizing Flows for Mobility Transfer},
 year = {2021}
}
```

## Contact
In case of any issues, please send a mail to
```guptavinayak51 (at) gmail (dot) com```
