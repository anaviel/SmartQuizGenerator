#!/bin/bash

cd ./quiz_gen_django

# Build the project
echo "Building the project..."
python3 -m pip install -r requirements.txt

echo "Collect Static..."
python3 manage.py collectstatic --noinput --clear