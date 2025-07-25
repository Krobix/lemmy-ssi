#!/usr/bin/env python3
"""
original script by vile asslips

• Posts once per `postfreq` hours, replies every loop.
• Generates a title, then feeds that title back into the model to create the body.
• Replies use only the original text as context (no echo).
• Empty‑prompt fallback handles BOS token insertion.
• Optional toxicity filter, UTF‑8 logging, robust to API shapes.
"""

import argparse
import logging
import time
from pathlib import Path
from types import MappingProxyType
import yaml
from bot_thread import BotThread

# ------------------------------------------------------------------ #
#  Entrypoint                                                      #
# ------------------------------------------------------------------ #
def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))

    log_dir = Path(cfg.get("log_dir","logs"))
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG if cfg.get("debug") else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(
            log_dir / f"asslips6_{int(time.time())}.log", encoding="utf-8"
        )]
    )
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    ))
    logging.getLogger().addHandler(console)

    threads = [
        BotThread(MappingProxyType(b), MappingProxyType(cfg))
        for b in cfg["bots"]
    ]
    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\n[CTRL‑C] Shutting down…")
        for t in threads:
            t.stop_event.set()
        for t in threads:
            t.join()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.yaml",
                    help="Path to YAML config")
    main(ap.parse_args().config)
