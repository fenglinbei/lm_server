#!/bin/bash

SERVICE_NAME=llm_server

while true; do
    HEALTH=$(docker inspect --format='{{json .State.Health.Status}}' $SERVICE_NAME)
    if [ "$HEALTH" != "\"healthy\"" ]; then
        echo "Container is not healthy, restarting..."
        docker restart $SERVICE_NAME
    fi
    if [ "$HEALTH" == "\"healthy\"" ]; then
        echo "Container is running."
    fi
    sleep 60
done