#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
from pyodh.odh import S3


data_dir = "/tmp/data"
export_dir = "./model_out"
odh = S3()
bucket, path = odh.parse_s3_url(os.environ["INPUT_DATA_LOCATION"])
odh.download_data(bucket=bucket, target_dir=data_dir, prefix=path, unpack=True)

#intect --train --train_dir test_data/num-dataset/train_data/ --arch_dir=$PWD/test_data/num-dataset/ --export
ret=subprocess.run(["intect", "--train", "--train_dir", os.path.join(data_dir, "train_data"), "--arch_dir", data_dir, "--export", "--export_dir", export_dir])
if ret.returncode==1:
    raise Exception("Failed to run intect")

bucket_out, prefix_out = odh.parse_s3_url(os.environ['OUTPUT_MODEL_LOCATION'])
odh.upload_data(bucket=bucket_out, source=export_dir, key_prefix=prefix_out)