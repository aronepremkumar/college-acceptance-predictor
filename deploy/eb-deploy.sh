#!/bin/bash
set -e

echo "Initializing EB application..."
eb init -p docker admission --region us-east-1

echo "Creating environment (this takes 3-5 mins)..."
eb create admission --instance_type t3.micro

echo "Deploying your app..."
eb deploy

echo "Opening in browser..."
eb open

echo "Done! Check status:"
eb status
eb health
