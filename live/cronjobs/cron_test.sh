#!/bin/bash
source /home/ubuntu/tornado-venv/bin/activate;
cd /home/ubuntu/tutela-app;
source init_env.sh;

python /home/ubuntu/tutela-app/live/cronjobs/test.py;
