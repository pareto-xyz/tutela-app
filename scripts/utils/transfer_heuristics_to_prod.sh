#! /bin/bash

rsync -azP ../../data/heuristics/*_processed_* ubuntu@$TORNADOPROD:/home/ubuntu/tutela-app/data/heuristics