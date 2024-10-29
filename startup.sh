#!/bin/sh

set -e

# Run database migrations
# echo "Running database migrations..."
# alembic -c ./budapp/alembic.ini upgrade head

# Start the application
echo "Starting the application..."
uvicorn "$APP_NAME.main:app" --host 0.0.0.0 --port "$APP_PORT"
