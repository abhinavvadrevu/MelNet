import boto3
import botocore
from botocore.exceptions import ClientError
import os
import glob

def upload_file(local_file_path, s3_file_path, bucket):
    s3_client = boto3.client(
        's3'
    )
    try:
        response = s3_client.upload_file(local_file_path, bucket, s3_file_path)
    except ClientError as e:
        print("ERROR WITH UPLOAD")
        print(e)
        return False
    return True

def check_if_s3_path_exists(bucket, s3_file_path):
    s3 = boto3.resource('s3')
    try:
        s3.Object(bucket, s3_file_path).load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            # The object does not exist.
            return False
        else:
            # Something else has gone wrong.
            raise e
    else:
        # The object does exist.
        return True

def upload_logs(run_name, s3_folder_name):
    local_file_wildcard = os.path.join('../logs/', run_name, '*')
    files = glob.glob(local_file_wildcard)
    for file in files:
        filename = os.path.basename(file)
        s3_path = os.path.join(s3_folder_name, 'logs', run_name, filename)
        print('Uploading %s' % s3_path)
        upload_file(file, s3_path, 'melnet-training-runs')

def upload_checkpoints(run_name, s3_folder_name):
    bucket = 'melnet-training-runs'
    local_file_wildcard = os.path.join('../chkpt/', run_name, '*')
    files = glob.glob(local_file_wildcard)
    for file in files:
        filename = os.path.basename(file)
        s3_path = os.path.join(s3_folder_name, 'chkpt', run_name, filename)
        exists = check_if_s3_path_exists(bucket, s3_path)
        if exists:
            print("Skipping upload of %s - it was already uploaded" % filename)
            continue
        print('Uploading %s' % filename)
        upload_file(file, s3_path, bucket)

run = 'blizzard-compressed-alldata-t1-tts'
upload_checkpoints(run, 'blizzard-compressed')
upload_logs(run, 'blizzard-compressed')