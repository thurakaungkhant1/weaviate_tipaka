import os
import weaviate
from weaviate.classes.init import Auth

# Best practice: store your credentials in environment variables
weaviate_url = os.environ["https://gzxjdpiatyodwtpwgoneq.c0.asia-southeast1.gcp.weaviate.cloud"]
weaviate_api_key = os.environ["eUVWaCttZXRMcDlERXFoQ19JOVZKTTdteVJOdWFGL0FmVGpwRUl1RnJZREJqVkhsNmVGWkNiNDM1L293PV92MjAw"]

# Connect to Weaviate Cloud
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,
    auth_credentials=Auth.api_key(weaviate_api_key),
)

print(client.is_ready())