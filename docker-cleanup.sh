#!/bin/bash

APP_NAME="team4s25-app"

# Check and stop running containers
RUNNING_CONTAINERS=$(docker ps -q --filter "name=$APP_NAME")
if [ -n "$RUNNING_CONTAINERS" ]; then
    echo "Stopping running containers for $APP_NAME..."
    docker stop $RUNNING_CONTAINERS
else
    echo "No running containers found for $APP_NAME."
fi

# Check and remove running/stopped containers
CONTAINERS=$(docker ps -a -q --filter "name=$APP_NAME")
if [ -n "$CONTAINERS" ]; then
    echo "Removing containers for $APP_NAME..."
    docker rm -f $CONTAINERS
else
    echo "No containers found for $APP_NAME."
fi

# Check and remove images
IMAGES=$(docker images -q "$APP_NAME")
if [ -n "$IMAGES" ]; then
    echo "Removing image $APP_NAME..."
    docker rmi -f $IMAGES
else
    echo "No image found for $APP_NAME."
fi

echo "Cleanup completed."
