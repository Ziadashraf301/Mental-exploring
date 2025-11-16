import tensorflow as tf
from src.logger.train_logger import get_logger
import os


def configure_gpu(config):
    """Configure GPU settings"""
    LOGGER = get_logger()
    if config.use_gpu:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Set memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set memory fraction
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=int(1024 * config.gpu_memory_fraction)
                    )]
                )
                LOGGER.info(f"GPU configured: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                LOGGER.warning(f"GPU configuration error: {e}")
        else:
            LOGGER.warning("No GPU found, using CPU")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        LOGGER.info("GPU disabled, using CPU")