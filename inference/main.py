import os
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
IMAGE_DIR = os.environ.get("IMAGE_DIR")  # Directory in S3 containing images

if not S3_BUCKET or not TENANT_NAME or not IMAGE_DIR:
    raise ValueError("S3_BUCKET, TENANT_NAME, and IMAGE_DIR must be set in the environment variables.")

s3 = boto3.client('s3')

def list_images_in_s3_directory(bucket, directory):
    """List all images in the specified S3 directory."""
    response = s3.list_objects_v2(Bucket=bucket, Prefix=directory)
    if 'Contents' not in response:
        return []
    return [item['Key'] for item in response['Contents'] if item['Key'].lower().endswith(('.jpg', '.jpeg', '.png'))]

def download_image_from_s3(bucket, s3_key, local_filename):
    """Download an image from S3 to a local file."""
    s3.download_file(bucket, s3_key, local_filename)
    return local_filename

def extract_features(image_path):
    """Extract features using ResNet50."""
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = model.predict(image)
    return features

def download_model(tenant_name):
    """Download the model artifact from S3."""
    artifact_path = f"{tenant_name}_features.npy"
    s3_key = f"{tenant_name}/artifacts/model_features.npy"
    s3.download_file(S3_BUCKET, s3_key, artifact_path)
    features = np.load(artifact_path)
    os.remove(artifact_path)
    return features

def recommend(query_features, all_features, top_n=5):
    """Find the top N similar items."""
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(query_features, all_features)
    indices = np.argsort(similarities[0])[::-1]
    return indices[:top_n]

if __name__ == "__main__":
    # List images in the specified S3 directory
    image_keys = list_images_in_s3_directory(S3_BUCKET, IMAGE_DIR)
    if not image_keys:
        raise ValueError(f"No images found in the directory {IMAGE_DIR} in bucket {S3_BUCKET}.")

    # Load tenant-specific model features from S3
    all_features = download_model(TENANT_NAME)

    # Process each image and recommend items
    for image_key in image_keys:
        local_image_path = "temp_image.jpg"
        download_image_from_s3(S3_BUCKET, image_key, local_image_path)
        try:
            # Extract features for the current image
            query_features = extract_features(local_image_path)

            # Recommend items for the current image
            recommendations = recommend(query_features, all_features)
            print(f"Recommended item indices for image {image_key}: {recommendations}")
        except Exception as e:
            print(f"Error processing image {image_key}: {e}")
        finally:
            # Clean up local image file
            os.remove(local_image_path)
