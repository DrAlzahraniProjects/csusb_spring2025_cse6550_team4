#!/bin/bash

APP_NAME="team4s25-app"
PORT=2504


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

# Check and release port if occupied
echo "Checking if port $PORT is in use..."
PORT_PID=$(lsof -ti:$PORT)
if [ -n "$PORT_PID" ]; then
    echo "Port $PORT is in use by process $PORT_PID. Killing process..."
    kill -9 $PORT_PID
    echo "Process killed. Port $PORT released."
else
    echo "Port $PORT is not in use."
fi


echo "Cleanup completed."