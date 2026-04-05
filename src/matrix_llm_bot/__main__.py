from __future__ import annotations

import argparse
import asyncio
import logging
import os

from .bot import MatrixLLMBot
from .config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description="Matrix LLM Bot")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument("--log-level", default=None, help="Logging level (default: INFO)")
    args = parser.parse_args()

    log_level = args.log_level or os.environ.get("LOG_LEVEL", "INFO")

    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = Config.load(args.config)
    bot = MatrixLLMBot(config)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
