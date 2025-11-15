import tensorflow as tf
from src.logger.train_logger import pipeline_logger
import os


def configure_gpu(config):
    """Configure GPU settings"""
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
                pipeline_logger.info(f"GPU configured: {len(gpus)} GPU(s) available")
            except RuntimeError as e:
                pipeline_logger.warning(f"GPU configuration error: {e}")
        else:
            pipeline_logger.warning("No GPU found, using CPU")
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        pipeline_logger.info("GPU disabled, using CPU")