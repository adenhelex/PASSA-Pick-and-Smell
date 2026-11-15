import logging

import uvicorn

from app.config import Config
from app.main import app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config.load()
    uvicorn.run(app, host=cfg.server_host, port=cfg.server_port, log_level="info")


if __name__ == "__main__":
    main()
