# Bluerred Image Detection
# 
# Author: Jasonsey
# Email: 2627866800@qq.com
# 
# =============================================================================
"""global config file"""
# ------------------------------net config------------------------
NUM_CLASS = 2                   # number of classes
CUDA_VISIBLE_DEVICES = '2'      # which device to use
BATCH_SIZE = 32                 # the batch size to train the CNN model
PREDICT_GPU_MEMORY = 0.08       # limit the gpu use for each process

# ------------------------------queue config-----------------------
THRIFT_HOST = '172.18.31.211'   # the host of thrift served
THRIFT_PORT = 9099              # the port of thrift served
THRIFT_NUM_WORKS = 12           # how manny subprocess the thrift use


def init_config():
    """config the remaining configuration"""
    global GPUS
    GPUS = len(CUDA_VISIBLE_DEVICES.split(','))


init_config()
