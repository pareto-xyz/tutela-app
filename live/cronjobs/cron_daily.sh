#!/bin/bash
source /home/ubuntu/tornado-venv/bin/activate;
cd /home/ubuntu/tutela-app;
source init_env.sh;

python /home/ubuntu/tutela-app/live/depositreuse/data.py;
python /home/ubuntu/tutela-app/live/tornadocash/data.py;
python /home/ubuntu/tutela-app/live/tornadocash/features.py;
