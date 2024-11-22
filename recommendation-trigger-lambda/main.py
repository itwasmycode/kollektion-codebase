import boto3
import os
import json

def lambda_handler(event, context):
    print(f"Received event: {json.dumps(event)}")

    batch_client = boto3.client("batch")

    # Extract environment variables from the event
    job_queue = os.getenv("JOB_QUEUE")
    job_definition = os.getenv("JOB_DEFINITION")

    # Dynamic inputs from the event
    job_name = event.get("jobName", "dynamic-batch-job")
    tenant_name = event.get("tenant_name")
    tenant_folder = event.get("tenant_folder")
    tenant_json = event.get("tenant_json")

    # Construct environment variables for Batch
    environment_vars = [
        {"name": "TENANT_NAME", "value": tenant_name},
        {"name": "TENANT_FOLDER", "value": tenant_folder},
        {"name": "TENANT_JSON", "value": tenant_json}
    ]

    # Submit the Batch job
    response = batch_client.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={
            "environment": environment_vars
        }
    )

    return {
        "statusCode": 200,
        "message": "Job submitted successfully.",
        "jobId": response["jobId"]
    }
