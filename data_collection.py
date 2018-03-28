from boto3 import session
from botocore.client import Config
import os


def upload(filename):
    # Initiate session
    path = os.path.abspath(filename)
    sessi = session.Session()

    client = sessi.client('s3',
                            region_name='nyc3',
                            endpoint_url='https://rit-missing-data.nyc3.digitaloceanspaces.com/',
                            aws_access_key_id=os.environ['SPACES_ACCESS'],
                            aws_secret_access_key=os.environ['SPACES_SECRET'])

    # Upload a file to your Space
    client.upload_file(path, 'experiment-results', filename)

def download(filename):
    # Initiate session
#    path = os.path.abspath(filename)
    sessi = session.Session()

    client = sessi.client('s3',
                            region_name='nyc3',
                            endpoint_url='https://rit-missing-data.nyc3.digitaloceanspaces.com/',
                            aws_access_key_id=os.environ['SPACES_ACCESS'],
                            aws_secret_access_key=os.environ['SPACES_SECRET'])

    # Upload a file to your Space
    client.download_file('experiment-results', filename, filename)
