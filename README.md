# End-to-End-Cars-Tanks-Classification using MLflow-DVC with AWS-CICD-Deployment-with-Github-Actions guide 

### Demo
https://huggingface.co/spaces/AbdallahAhmed98/CarsTanksClassification

### MLflow repo
https://dagshub.com/Abdallah-Ahmed-98/Cars-Tanks-Classification

### MLflow UI
https://dagshub.com/Abdallah-Ahmed-98/Cars-Tanks-Classification.mlflow

## Create and activate virtual environment

```bash
conda create -n CTCls python=3.10 -y
```
```bash
conda activate CTCls
```

## Install requirements

```bash
pip install -r requirements.txt
```



## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

### You have to provide yours not mine!

MLFLOW_TRACKING_URI=https://dagshub.com/Abdallah-Ahmed-98/Cars-Tanks-Classification.mlflow \
MLFLOW_TRACKING_USERNAME=Abdallah-Ahmed-98 \
MLFLOW_TRACKING_PASSWORD=************************************** \

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/Abdallah-Ahmed-98/Cars-Tanks-Classification.mlflow

export MLFLOW_TRACKING_USERNAME=Abdallah-Ahmed-98 

export MLFLOW_TRACKING_PASSWORD=**************************************

```



### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

