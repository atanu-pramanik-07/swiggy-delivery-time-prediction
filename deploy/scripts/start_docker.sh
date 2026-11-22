#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 273907458507.dkr.ecr.ap-southeast-2.amazonaws.com

echo "Pulling Docker image..."
docker pull 273907458507.dkr.ecr.ap-southeast-2.amazonaws.com/swiggy-delivery-time-prediction:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=delivery_time_pred)" ]; then
    echo "Stopping existing container..."
    docker stop delivery_time_pred
fi

if [ "$(docker ps -aq -f name=delivery_time_pred)" ]; then
    echo "Removing existing container..."
    docker rm delivery_time_pred
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name delivery_time_pred -e DAGSHUB_USER_TOKEN=40bbc9c6b932e0185657eeb2ab55918d65633cd7 273907458507.dkr.ecr.ap-southeast-2.amazonaws.com/swiggy-delivery-time-prediction:latest

echo "Container started successfully."