import os
import tarfile
import boto3
from urllib.parse import urlparse
import logging

_LOGGER = logging.getLogger(name=__name__)

class S3(object):
    def __init__(self, url=None, key=None, id=None):
        self.conf = {
            "url": None,
            "key": None,
            "id": None
        }
        self.load_conf(url, key, id)
        self.conn = self.connect()

    def load_conf(self, url=None, key=None, id=None):

        if os.environ.get("S3_ENDPOINT_URL"):
            self.conf['url'] = os.environ.get("S3_ENDPOINT_URL")
        else:
            self.conf['url'] = url

        if os.environ.get("AWS_SECRET_ACCESS_KEY"):
            self.conf['key'] = os.environ.get("AWS_SECRET_ACCESS_KEY")
        else:
            self.conf['key'] = key

        if os.environ.get("AWS_ACCESS_KEY_ID"):
            self.conf['id'] = os.environ.get("AWS_ACCESS_KEY_ID")
        else:
            self.conf['id'] = id


    def connect(self):
        conn = boto3.client(service_name='s3',
            aws_access_key_id=self.conf['id'],
            aws_secret_access_key=self.conf['key'],
            endpoint_url=self.conf['url'])
        return conn

    def download_data(self, bucket, target_dir=None, prefix=None, unpack=False):
        for f in self.conn.list_objects(Bucket=bucket, Prefix=prefix)['Contents']:
            target_file = f['Key'].split("/")[-1]
            if target_dir:
                if not os.path.isdir(target_dir):
                    os.mkdir(target_dir, 0o775)
                target_file = os.path.join(target_dir, target_file)

            _LOGGER.warn("Downloading %s from %s" % (target_file, f['Key']))
            self.conn.download_file(Bucket=bucket, Key=f['Key'], Filename="%s" % target_file)
            if unpack:
                tar = tarfile.open(target_file)
                tar.extractall(path=target_dir)
                tar.close()

    def upload_data(self, bucket, source=None, key_prefix=None):
        if not source:
            source = "."
        
        files = source
        if isinstance(files, str):
            if os.path.isdir(source):
                output_file = "%s.tgz" % source.split("/")[-1]
                self._make_tarfile(output_file, source)
                files = [output_file]
            else:
                files = [source]

        for f in files:
            self.conn.upload_file(Bucket=bucket, Key=os.path.join(key_prefix, f), Filename=f)
            print("File %s uploaded" % f)

    import tarfile

    def _make_tarfile(self, output_filename, source_dir):
        _LOGGER.warn("Creating gzipped tar %s from %s" %(output_filename, source_dir))
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))

    def parse_s3_url(self, url):
        o = urlparse(url)
        return o.netloc, o.path