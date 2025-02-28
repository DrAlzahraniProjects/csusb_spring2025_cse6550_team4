#!/bin/bash

# Define container name
CONTAINER_NAME="team4s25-app"

# Stop the running container
echo "Stopping Docker container..."
docker stop $CONTAINER_NAME

# Remove the container
echo "Removing Docker container..."
docker rm $CONTAINER_NAME

# Remove the Docker image
echo "Removing Docker image..."
docker rmi $CONTAINER_NAME

echo "Cleanup complete."