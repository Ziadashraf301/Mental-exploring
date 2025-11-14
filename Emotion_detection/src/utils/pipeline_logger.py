import logging
from pathlib import Path

# Create logs directory if not exists
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)

# Log format
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

# Configure basic logging (writes to file)
logging.basicConfig(
    filename=LOG_DIR / "pipeline.log",
    level=logging.INFO,
    format=LOG_FORMAT
)

# Create global logger
pipeline_logger = logging.getLogger("pipeline_logger")
pipeline_logger.setLevel(logging.INFO)

# Add console handler (optional but recommended)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

pipeline_logger.addHandler(console_handler)
