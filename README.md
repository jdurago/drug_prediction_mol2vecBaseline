# Drug Prediction
Implementation of Mol2Vec with a random forest classifier for the [HIV MoleculeNet dataset](http://moleculenet.ai/datasets-1). This package is meant to be run as an AWS Sagemaker training job. This is a baseline model from which more complicated models such as a graph neural nets will be compared to.

# Build Model and Train
Sagemaker training jobs utilize docker containers stored in AWS Elastic Container Registry to execute. Code for running a local and container based training jobs can be found in the <b> run_container.ipynb </b>

Steps for running a training job are as follows:
1. Build container and upload to AWS ECR using the build_and_push.sh script <br />
    `!cd container; ./build_and_push.sh drug-prediction`
2. To run a local training job <br />

    <b>Download Datasets:</b> <br />
    <b> a. Pretrained mol2vec model </b> <br />
    `wget -O data/model_300dim.pkl https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl model_300dim.pkl`

    <b> b. Example data</b> <br />  
    `wget -O data/ames.sdf https://github.com/samoturk/mol2vec/raw/master/examples/data/ames.sdf`

    <b> c. MoleculeNet HIV data</b> <br />
    `wget -O data/hiv.zip https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/hiv.zip`
    `cd data; unzip hiv.zip`
    
    <b>Within Python run the following: </b>
    ```python
    from sagemaker import get_execution_role
    role = get_execution_role()
    
    import os
    import subprocess

    instance_type = 'local'
    if subprocess.call('nvidia-smi') == 0:
        ## Set type to GPU if one is present
        instance_type = 'local_gpu'
    print("Instance type = " + instance_type)
    
    from sagemaker.estimator import Estimator
    hyperparameters = {'dev-mode': True}
    estimator = Estimator(role=role,
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          image_name='drug-prediction:latest',
                          hyperparameters=hyperparameters)

    estimator.fit(<PATH TO LOCAL DATA>)
    ```
    <br />

3. To run a Sagemaker training job <br />
    ```python
    # Get Sagemaker session
    import sagemaker as sage
    sess = sage.Session()
    
    # Get ECR Image Info
    import boto3
    client = boto3.client('sts')
    account = client.get_caller_identity()['Account']

    my_session = boto3.session.Session()
    region = my_session.region_name
    algorithm_name = 'drug-prediction'
    ecr_image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, algorithm_name)
    print(ecr_image)
    
    from sagemaker.estimator import Estimator

    data_location = 's3://<Insert S3 Path to Data>'
    output_path = 's3://<Insert Desired S3 Path for model and scoring>'
    max_run_time = 3*60*60 # train for max of 3 hours
    hyperparameters = {'dev-mode': False} # False uses actual dataset

    instance_type = 'ml.m5.4xlarge' # use m5.4xlarge instance

    estimator = Estimator(role=role,
                          train_instance_count=1,
                          train_instance_type=instance_type,
                          image_name=ecr_image,
                          hyperparameters=hyperparameters,
                         output_path = output_path,
                         train_max_run=max_run_time)

    estimator.fit(data_location)
    ```

