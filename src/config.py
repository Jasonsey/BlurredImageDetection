"""
config file
"""
# ------------------------------net config------------------------
NUM_CLASS = 2
CUDA_VISIBLE_DEVICES = '2'
BATCH_SIZE = 32
PREDICT_GPU_MEMORY = 0.08

# ------------------------------redis config------------------------
REDIS_HOST = '172.18.31.211'
REDIS_PORT = '6379'
REDIS_DB = 0
REDIS_PASSWD = '123456'
IMAGE_QUEUE = 'blur_detection'

# ------------------------------queue config-----------------------
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# ------------------------------queue config-----------------------
THRIFT_HOST = '172.18.31.211'
THRIFT_PORT = 9099
THRIFT_NUM_WORKS = 12


def init_config():
    """
    通过现有配置计算剩余的配置值
    """
    global GPUS
    GPUS = len(CUDA_VISIBLE_DEVICES.split(','))


init_config()
