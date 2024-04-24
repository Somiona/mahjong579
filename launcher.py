import argparse
import asyncio
import logging
import os
import shutil
from pathlib import Path

from data.downloader import download_data, download_logs, read_scc
from game import Server


def clear_tenhou_data():
    shutil.rmtree(data_dir)
    shutil.rmtree(log_dir)
    os.makedirs(data_dir)
    os.makedirs(log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--load_scc",
        action="store_true",
        help="This option loads data from tenhou_scc folder",
    )
    parser.add_argument(
        "-r",
        "--reload_data",
        action="store_true",
        help="This open clears both tenhou_logs and tenhou_data folders",
    )
    parser.add_argument("--min_score", "-m", default=0, type=int)
    parser.add_argument(
        "--fast", "-f", action="store_true", help="Cancel AI thinking time"
    )
    parser.add_argument("--train", "-t", action="store_true", help="Collect playing data")
    args = parser.parse_args()

    if args.reload_data and args.load_scc:
        raise ValueError("Cannot reload data from tenhou and load scc at the same time")

    project_dir = Path(__file__).parent
    data_dir = project_dir / "train_data"
    log_dir = project_dir / "tenhou_logs"
    scc_dir = project_dir / "tenhou_scc"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(scc_dir, exist_ok=True)

    if args.reload_data:
        clear_tenhou_data()
        download_logs()
        download_data()

    if args.load_scc:
        assert os.path.isdir(scc_dir), f"Path not exist: {scc_dir}"
        assert os.listdir(scc_dir), f"Path is empty: {scc_dir}"
        clear_tenhou_data()
        read_scc()
        # move all files from scc_dir to log_dir
        for file in os.listdir(scc_dir):
            if file.startswith("scc") and file.endswith(".txt"):
                shutil.move(scc_dir / file, log_dir / file)
        download_data()

    logging.basicConfig(level=logging.DEBUG)
    server = Server(args.min_score, args.fast, args.train)
    asyncio.run(server.run())