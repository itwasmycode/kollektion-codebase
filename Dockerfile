# Start from a Python 3.12 image
FROM public.ecr.aws/lambda/python:3.12

# Set the working directory.
WORKDIR /var/task

# Copy requirements.txt to the container
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory to the container
COPY . ./
