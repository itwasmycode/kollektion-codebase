# Start from a Python 3.12 image
FROM python:3.12-slim
# Set the working directory inside the container
WORKDIR /app
# Copy requirements.txt to the container
COPY requirements.txt .
# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the entire project directory to the container
COPY . /app
