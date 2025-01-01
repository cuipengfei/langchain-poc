# logging_setup.py
import http.client
import logging


def setup_logging() -> None:
    logging.basicConfig(
        format="%(levelname)s [%(asctime)s] %(name)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO
    )
    httpclient_logger: logging.Logger = logging.getLogger("http.client")

    def httpclient_log(*args: str) -> None:
        httpclient_logger.log(logging.DEBUG, " ".join(args))

    http.client.print = httpclient_log
    http.client.HTTPConnection.debuglevel = 1

    urllib3_logger: logging.Logger = logging.getLogger("urllib3")
    urllib3_logger.setLevel(logging.DEBUG)

    for handler in urllib3_logger.handlers:
        handler.setLevel(logging.DEBUG)
