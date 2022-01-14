# Notes on installation

Need google.cloud. Install instructions [here](https://cloud.google.com/sdk/docs/install) or follow these steps. This only needs to be done once on a new computer / ec2 instance.
```
# installs gcloud and gsutil command line utils
wget https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-368.0.0-linux-x86_64.tar.gz .
tar -xzvf google-cloud-sdk-368.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
./google-cloud-sdk/bin/gcloud init  # will require browser auth
```

```
# install python utilities
# make sure you are in the virtual env
pip install --upgrade google-cloud-storage
pip install --upgrade google-cloud-bigqueryls
```

Then `scp` your credentials over to a save space on the instance and then define `GOOGLE_APPLICATION_CREDENTIALS` in `~/.bashrc`.
