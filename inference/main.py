import os
import json
import boto3
import numpy as np
import psycopg2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics.pairwise import cosine_similarity

# Initialize ResNet50
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

# AWS and RDS Configuration
S3_BUCKET = os.environ.get("S3_BUCKET")
TENANT_NAME = os.environ.get("TENANT_NAME")
IMAGE_DIR = os.environ.get("IMAGE_DIR")
RDS_HOST = os.environ.get("RDS_HOST")
RDS_PORT = os.environ.get("RDS_PORT", 5432)
RDS_USER = os.environ.get("RDS_USER")
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")
RDS_DATABASE = os.environ.get("RDS_DATABASE")
VERSION = os.environ.get("VERSION")

if not all([S3_BUCKET, TENANT_NAME, IMAGE_DIR, RDS_HOST, RDS_USER, RDS_PASSWORD, RDS_DATABASE, VERSION]):
    raise ValueError("All required environment variables must be set.")

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

def download_category_compatibility(s3_bucket, tenant_name):
    """Download category compatibility JSON from S3."""
    compatibility_file = f"{tenant_name}/artifacts/category_compability.json"
    local_file = "category_compability.json"
    s3.download_file(s3_bucket, compatibility_file, local_file)
    with open(local_file, "r") as f:
        category_compatibility = json.load(f)
    os.remove(local_file)
    return category_compatibility

def connect_to_rds():
    """Connect to RDS."""
    try:
        return psycopg2.connect(
            host=RDS_HOST,
            port=RDS_PORT,
            user=RDS_USER,
            password=RDS_PASSWORD,
            database=RDS_DATABASE
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to RDS: {e}")

def create_table_if_not_exists(conn):
    """Create the Recommendations table if it does not exist."""
    create_table_query = '''
        CREATE TABLE IF NOT EXISTS Recommendations (
            customer_id UUID NOT NULL,
            item_id INT NOT NULL,
            recommendations JSONB NOT NULL,
            version VARCHAR(20) NOT NULL,
            revision INT DEFAULT 1,
            PRIMARY KEY (customer_id, item_id)
        );
    '''
    try:
        with conn.cursor() as cursor:
            cursor.execute(create_table_query)
            conn.commit()
            print("Table 'Recommendations' is ready.")
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to create table: {e}")

def get_customer_id(conn, tenant_name):
    """Fetch customer_id from the cust_map table using TENANT_NAME."""
    query = '''
    SELECT ext_cust_id
    FROM cust_map
    WHERE cust_name = %s;
    '''
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, (tenant_name,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"No customer_id found for tenant_name: {tenant_name}")
            return result[0]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch customer_id: {e}")

def write_recommendations_to_rds(conn, customer_id, item_id, recommendations, version):
    """Write recommendations to RDS."""
    formatted_recommendations = json.dumps(recommendations)  # Format collections as JSON

    sql = '''
    INSERT INTO Recommendations (customer_id, item_id, recommendations, version, revision)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (customer_id, item_id) DO UPDATE
    SET recommendations = EXCLUDED.recommendations,
        version = EXCLUDED.version,
        revision = Recommendations.revision + 1;
    '''
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, (customer_id, item_id, formatted_recommendations, version, 1))
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to write recommendations: {e}")


def recommend(query_features, all_features, mapping, query_category, category_compatibility, num_collections=5, collection_size=5):
    """Generate collections ensuring unique categories and diverse items per collection."""
    compatible_categories = category_compatibility.get(query_category, [])
    indices = [i for i, item in enumerate(mapping) if any(cat in compatible_categories for cat in item["categories"])]

    # Exit early if no compatible items are found
    if not indices:
        print(f"No compatible items found for category '{query_category}'. Skipping recommendations.")
        return {}

    compatible_features = all_features[indices]
    similarities = cosine_similarity(query_features, compatible_features)

    # Group items by category
    category_groups = {category: [] for category in compatible_categories}
    for i in indices:
        item = mapping[i]
        for category in item["categories"]:
            if category in compatible_categories:
                category_groups[category].append(item)

    # Generate collections
    collections = {}
    used_items_per_category = {category: set() for category in compatible_categories}

    for collection_id in range(1, num_collections + 1):
        collection = []
        used_categories = set()

        for category in compatible_categories:
            available_items = [
                item for item in category_groups[category]
                if item["id"] not in used_items_per_category[category]
            ]

            # If no unused items are left in this category, skip
            if not available_items:
                continue

            selected_item = available_items[0]  # Select the first available item
            collection.append(selected_item)
            used_items_per_category[category].add(selected_item["id"])
            used_categories.add(category)

            if len(collection) == collection_size:
                break

        # Ensure diversity in subsequent collections
        if collection_id > 1 and collection:
            previous_collection = collections.get(str(collection_id - 1), [])
            if all(item in previous_collection for item in collection):
                for category in compatible_categories:
                    alternate_items = [
                        item for item in category_groups[category]
                        if item not in collection
                    ]
                    if alternate_items and collection:
                        collection[-1] = alternate_items[0]
                        break


        # Add collection only if it meets the minimum size
        if collection:
            collections[str(collection_id)] = collection

    return collections


if __name__ == "__main__":
    conn = connect_to_rds()
    try:
        # Load features and mapping
        s3.download_file(S3_BUCKET, f"{TENANT_NAME}/artifacts/{TENANT_NAME}_features.npy", "features.npy")
        s3.download_file(S3_BUCKET, f"{TENANT_NAME}/artifacts/mapping.json", "mapping.json")
        all_features = np.load("features.npy")
        with open("mapping.json", "r") as f:
            mapping = json.load(f)

        # Download category compatibility
        category_compatibility = download_category_compatibility(S3_BUCKET, TENANT_NAME)

        # Fetch customer_id from cust_map table
        customer_id = get_customer_id(conn, TENANT_NAME)

        for image_key in list_images_in_s3_directory(S3_BUCKET, IMAGE_DIR):
            local_image_path = "temp_image.jpg"
            download_image_from_s3(S3_BUCKET, image_key, local_image_path)
            query_features = extract_features(local_image_path)
            current_item = next((item for item in mapping if image_key in item["images"]), None)
        
            if not current_item:
                print(f"Warning: Item for key '{image_key}' not found in mapping.")
                continue
        
            collections = recommend(
                query_features, all_features, mapping,
                current_item["categories"][0],
                category_compatibility
            )
            
            if collections:
                create_table_if_not_exists(conn)
                write_recommendations_to_rds(conn, customer_id, current_item["id"], collections, VERSION)
            else:
                print(f"No recommendations generated for item ID {current_item['id']}.")

    finally:
        conn.close()
