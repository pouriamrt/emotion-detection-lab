import sys
from loguru import logger
from src.config import LOGS_DIR

logger.remove()

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[user]:^20}</cyan> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

logger.add(sys.stderr, level="INFO", format=LOG_FORMAT,
           filter=lambda record: record["extra"].setdefault("user", "system") or True)

logger.add(str(LOGS_DIR / "app.log"), rotation="10 MB", retention="30 days",
           level="DEBUG", format=LOG_FORMAT,
           filter=lambda record: record["extra"].setdefault("user", "system") or True)

logger.add(str(LOGS_DIR / "auth.log"), rotation="5 MB", retention="30 days",
           level="INFO", format=LOG_FORMAT,
           filter=lambda record: "auth" in record["extra"].get("context", ""))

logger.add(str(LOGS_DIR / "training.log"), rotation="10 MB", retention="30 days",
           level="INFO", format=LOG_FORMAT,
           filter=lambda record: "training" in record["extra"].get("context", ""))


def get_logger(user: str = "system", context: str = "general"):
    return logger.bind(user=user, context=context)
