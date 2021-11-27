import redis

rds = redis.Redis(host='localhost', port=6380, db=0)
rds.flushdb()

