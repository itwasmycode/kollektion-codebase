# Start from a Python 3.12 image
FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt to the container
COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

# Copy the entire project directory to the container
COPY . ${LAMBDA_TASK_ROOT}
