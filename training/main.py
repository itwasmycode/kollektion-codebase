import os
import json
import boto3
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests

# Initialize ResNet50
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# AWS Configuration
S3_BUCKET = os.environ.get("S3_BUCKET")
TENANT_NAME = os.environ.get("TENANT_NAME")
DYNAMO_TABLE = os.environ.get("DYNAMO_TABLE")  # DynamoDB table for versioning
REGION = os.environ.get("REGION")

if not S3_BUCKET or not TENANT_NAME or not DYNAMO_TABLE:
    raise ValueError("S3_BUCKET, TENANT_NAME, and DYNAMO_TABLE must be set in the environment variables.")

s3_client = boto3.client('s3')
dynamo_client = boto3.client('dynamodb',
                             region=REGION)

def increment_version(dynamo_table):
    """Increment version number stored in DynamoDB."""
    try:
        response = dynamo_client.update_item(
            TableName=dynamo_table,
            Key={"key": {"S": "version"}},
            UpdateExpression="SET #val = if_not_exists(#val, :start) + :incr",
            ExpressionAttributeNames={"#val": "value"},
            ExpressionAttributeValues={":start": {"N": "0"}, ":incr": {"N": "1"}},
            ReturnValues="UPDATED_NEW"
        )
        return response["Attributes"]["value"]["N"]
    except Exception as e:
        raise RuntimeError(f"Failed to update version in DynamoDB: {e}")

def extract_features(image_path):
    """Extract features using ResNet50."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return model.predict(image)

def process_json(s3_key, tenant_name, dynamo_table):
    """Process the JSON file from S3 and store features and mapping in S3."""
    version = increment_version(dynamo_table)
    print(f"Processing version: {version}")

    local_json_file = "temp.json"
    s3_client.download_file(S3_BUCKET, s3_key, local_json_file)

    with open(local_json_file, "r") as f:
        data = json.load(f)

    features, mapping = [], []
    for item in data:
        image_url = item["images"][0]
        image_filename = os.path.basename(image_url)
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            image_path = "temp.jpg"
            with open(image_path, "wb") as img_file:
                img_file.write(response.content)
            feature_vector = extract_features(image_path)
            features.append(feature_vector)
            mapping.append({
                "id": item["id"],
                "categories": item["categories"],
                "images": item["images"]
            })
            os.remove(image_path)

    features = np.vstack(features)
    np.save(f"{tenant_name}_features.npy", features)
    with open("mapping.json", "w") as f:
        json.dump(mapping, f)

    s3_client.upload_file(f"{tenant_name}_features.npy", S3_BUCKET, f"{tenant_name}/artifacts/{tenant_name}_features.npy")
    s3_client.upload_file("mapping.json", S3_BUCKET, f"{tenant_name}/artifacts/mapping.json")

    os.remove(f"{tenant_name}_features.npy")
    os.remove("mapping.json")
    os.remove(local_json_file)

if __name__ == "__main__":
    s3_key = os.environ.get("JSON_S3_KEY")
    if not s3_key:
        raise ValueError("JSON_S3_KEY must be set in the environment variables.")
    process_json(s3_key, TENANT_NAME, DYNAMO_TABLE)
