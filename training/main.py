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

# AWS S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET")
TENANT_NAME = os.environ.get("TENANT_NAME")

if not S3_BUCKET or not TENANT_NAME:
    raise ValueError("S3_BUCKET and TENANT_NAME must be set in the environment variables.")

s3_client = boto3.client('s3')

def download_image_and_upload_to_s3(url, tenant_name, filename):
    """Download image from a URL and upload to S3."""
    tenant_images_s3_path = f"{tenant_name}_images/{filename}"
    try:
        response = requests.get(url, timeout=10)  # Add a timeout to avoid hanging
        if response.status_code == 200:
            s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=tenant_images_s3_path,
                Body=response.content,
                ContentType="image/jpeg"
            )
            print(f"Image uploaded to S3: {tenant_images_s3_path}")
            return tenant_images_s3_path
        else:
            print(f"Failed to download image. HTTP status code: {response.status_code}")
    except requests.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
    return None  # Return None if the download fails

def extract_features(image_path):
    """Extract features using ResNet50."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features

def download_json_from_s3(bucket, s3_key, local_file):
    """Download a JSON file from S3."""
    s3_client.download_file(bucket, s3_key, local_file)
    print(f"Downloaded {s3_key} from S3 to {local_file}")
    return local_file

def process_json(s3_key, tenant_name):
    """Process the JSON file from S3 and store the model in S3."""
    # Download JSON from S3
    local_json_file = "temp.json"
    download_json_from_s3(S3_BUCKET, s3_key, local_json_file)

    # Load JSON data
    with open(local_json_file, "r") as f:
        data = json.load(f)

    # Extract features for all images
    all_features = []
    for item in data:
        image_url = item["images"][0]  # Use the first image
        image_filename = os.path.basename(image_url)
        image_s3_key = download_image_and_upload_to_s3(image_url, tenant_name, image_filename)
        if image_s3_key:  # Only process if the image was uploaded successfully
            try:
                # Download the image from S3 to a local temporary file
                local_image_path = "temp.jpg"
                s3_client.download_file(S3_BUCKET, image_s3_key, local_image_path)

                # Extract features
                features = extract_features(local_image_path)
                all_features.append(features)

                # Clean up local temporary file
                os.remove(local_image_path)
            except Exception as e:
                print(f"Error extracting features for {image_url}: {e}")
        else:
            print(f"Skipping image {image_url} due to download failure.")

    # Convert features to numpy array
    if all_features:
        all_features = np.vstack(all_features)

        # Save features to a local file
        artifact_path = f"{tenant_name}_features.npy"
        np.save(artifact_path, all_features)

        # Upload the artifact to S3
        artifact_s3_key = f"{tenant_name}/artifacts/model_features.npy"
        s3_client.upload_file(artifact_path, S3_BUCKET, artifact_s3_key)
        os.remove(artifact_path)

        print(f"Features saved and uploaded to S3: {artifact_s3_key}")
    else:
        print("No features extracted. Skipping model upload.")

    os.remove(local_json_file)


if __name__ == "__main__":
    # Get environment variables
    s3_key = os.environ.get("JSON_S3_KEY")  # The S3 key of the JSON file
    if not s3_key:
        raise ValueError("JSON_S3_KEY must be set in the environment variables.")

    process_json(s3_key, TENANT_NAME)
