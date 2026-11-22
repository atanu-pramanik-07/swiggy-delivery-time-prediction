#!/bin/bash
set -e

# Ensure non-interactive mode
export DEBIAN_FRONTEND=noninteractive

# Update package lists and install Docker + utilities
sudo apt-get update -y
sudo apt-get install -y docker.io unzip curl

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Install AWS CLI v2 only if not already installed (keeps re-deployments fast)
if ! command -v aws &> /dev/null; then
    curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
    unzip -qo /home/ubuntu/awscliv2.zip -d /home/ubuntu/
    sudo /home/ubuntu/aws/install --update
    rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws
fi

# Add 'ubuntu' user to the 'docker' group
sudo usermod -aG docker ubuntu