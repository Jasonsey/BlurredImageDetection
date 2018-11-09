import redis

import config


db = redis.StrictRedis(
    host = config.REDIS_HOST
    port=settingsc.REDIS_PORT,
    db=settingsc.REDIS_DB
)