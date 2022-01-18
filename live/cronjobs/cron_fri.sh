#!/bin/bash
source /home/ubuntu/tornado-venv/bin/activate;
cd /home/ubuntu/tutela-app;
source init_env.sh;

python live/tornadocash/heuristic.py --heuristic 5
