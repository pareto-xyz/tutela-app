#! /bin/bash

redis-server --port 6380 --maxmemory 1gb --maxmemory-policy allkeys-lru
