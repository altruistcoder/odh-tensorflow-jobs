#!/usr/bin/env python3

import os
from pyodh.odh import S3
import logging

odh = S3()
bucket, path = odh.parse_s3_url(os.environ["INPUT_MODEL_LOCATION"])
logging.warn("Using bucket %s and prefix %s", bucket, path)
odh.download_data(bucket=bucket, target_dir="./", prefix=path, unpack=True)
