#!/bin/bash
# Comprehensive Docker cleanup script for professor
# This will forcefully remove all instances of the team4s25 application

# Set the container and image names
APP_NAME="team4s25-app"
CONTAINER_PATTERN="team4s25"  # Broader pattern to catch any related containers

echo "Starting thorough cleanup of all team4s25 Docker resources..."

# 1. List all containers (for reference before removal)
echo "Current containers matching pattern '$CONTAINER_PATTERN':"
docker ps -a | grep $CONTAINER_PATTERN

# 2. Stop all containers matching the pattern (regardless of state)
echo "Stopping all matching containers..."
docker stop $(docker ps -a -q --filter name=$CONTAINER_PATTERN) 2>/dev/null || true

# 3. Force remove all containers matching the pattern
echo "Removing all matching containers..."
docker rm -f $(docker ps -a -q --filter name=$CONTAINER_PATTERN) 2>/dev/null || true

# 4. Remove specific image if it exists
echo "Removing specific image $APP_NAME if it exists..."
docker rmi -f $APP_NAME 2>/dev/null || true

# 5. Find and remove any images created from the Dockerfile (even if renamed)
echo "Removing any images created in the last 30 days with 'team4s25' in their metadata..."
docker images --format "{{.ID}}" --filter "reference=*team4s25*" | xargs -r docker rmi -f 2>/dev/null || true

# 6. Clean up any dangling images and volumes
echo "Cleaning up any dangling Docker resources..."
docker system prune -f

# 7. Verify cleanup was successful
REMAINING=$(docker ps -a | grep $CONTAINER_PATTERN | wc -l)
if [ "$REMAINING" -eq "0" ]; then
    echo "✅ All team4s25 containers successfully removed."
else
    echo "⚠️ Warning: $REMAINING containers still exist. Using stronger measures..."
    
    # Try with different filters and approaches
    docker ps -a | grep $CONTAINER_PATTERN | awk '{print $1}' | xargs -r docker rm -f
    
    # Check again
    REMAINING=$(docker ps -a | grep $CONTAINER_PATTERN | wc -l)
    if [ "$REMAINING" -eq "0" ]; then
        echo "✅ All containers successfully removed after additional steps."
    else
        echo "❌ Some containers could not be removed automatically."
        echo "Remaining containers:"
        docker ps -a | grep $CONTAINER_PATTERN
    fi
fi



echo "Cleanup process complete. You can now proceed with new deployment."