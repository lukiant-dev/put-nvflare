## Intro

This project is a showcase of the federated learning framework Nvidia Flare for the academic learning purposes. It's not production ready, it demonstrates some of the federated learning concepts, it provides an entrypoint for further learning and research. 

It's a ready to be executed sample project setup with two example jobs. 
Infrastructure is provided as the docker compose, so docker engine and docker compose are required on the machine. It was tested with 3cpu and 16gb allocated for docker. Probably xgboost should work with less resources, while cifar10 probably will not. 

## Repository structure

The project was created by first installing nvflare on the local filesystem then modifying default project.yml of nvflare with adding DockerBuilder section and launching `nvflare provision` command. 

This command creates the workspace with all the files that NVFlare needs to work. Each runtime gets its own directory with the certificates and configuration needed for running corresponding component. Additionally every user defined in project.yml gets directory with a user starter pack. 
In the production system each directory would be shipped to different site and used in the remote location. 

The compose.yaml was supplemented with tensorboard service to visualize accuracy progress of training jobs.

On top of this working infrastructure there are some minor usability enhancements added.

Last the two examples were added to show different nvflare jobs:
1) Needs downloading the dataset and copying it to transfer/cifar10_splits directory on each site:
cifar10_fedavg -> deviated from: https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/cifar10/cifar10-real-world

2) Ready to be used: 
uncover-xgboost -> build from: https://github.com/NVIDIA/NVFlare/tree/main/examples/advanced/xgboost/tree-based but with dataset coming from kaggle competition: https://www.kaggle.com/datasets/roche-data-science-coalition/uncover/data


## Execution 

1) Setup infrastructure:

```
docker compose build
docker compose up -d 
```

2) Run example:

```
cd workspace/put_project/prod_00/admin@nvidia.com/startup  
./fl_admin.sh
# user name is admin@nvidia.com
```

This opens the nvflare cli admin prompt. Try available commands with ? 
execute example with: 

```
submit_job uncover-xgboost
```

That should respond with job id like this:
```
Submitted job: b7b2736d-81ac-4524-97e4-be040211d437
Done [142970 usecs] 2025-01-17 14:42:57.814667
```

```
docker compose logs
```
is a helpful command that will show you logs from different containers. 


access localhost:6006 to see the training job accuracy metrics. 

