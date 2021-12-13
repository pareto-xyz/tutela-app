#! /bin/bash

rsync -azP ../../data/heuristics/*_processed_* ubuntu@$TORNADODEV:/home/ubuntu/tutela-app/data/heuristics