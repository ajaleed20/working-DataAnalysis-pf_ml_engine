import os
import yaml
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

def load_config():
    dir_root = os.path.dirname(os.path.abspath(__file__))
    with open(dir_root + "/config.yaml", "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.FullLoader)

config = load_config()
print(*config)

def get_files(dir):
    with os.scandir(dir) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                yield entry


def upload(files, connection_string, container_name):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    print("upload is in progress")

    for file in files:
        blob_client = container_client.get_blob_client(file.name)
        with open(file.path, "rb") as data:
            blob_client.upload_blob(data)
            print(f'{file.name} uploaded to blob storage')
            os.remove(file)
            print(f'{file.name} removed from folder')

try:
    print("Azure Blob Storage v" + __version__ + " - Python quickstart sample")
    config = load_config()
    csv_files = get_files(config["source_folder"] + "/csvfiles")
    upload(csv_files, config["azure_storage_connectionstring"], config["csv_container_name"])

except Exception as ex:
    print('Exception:')
    print(ex)
