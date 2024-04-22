import gzip
import os
import re
import shutil
import signal
from multiprocessing import Pool
from pathlib import Path

import requests
import tqdm

project_dir = os.path.abspath(Path(os.path.dirname(__file__)).parent)
log_dir = os.path.join(project_dir, "tenhou_logs")
data_dir = os.path.join(project_dir, "train_data")
scc_dir = os.path.join(project_dir, "tenhou_scc")
headers = {
    "Host": "tenhou.net",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.6 Safari/605.1.1",
}


def fetch_data(url: str) -> None:
    file_name = url.split("?")[1] + ".xml"
    try:
        response = requests.get(url, stream=True, headers=headers)
        with open(os.path.join(data_dir, file_name), "w") as out_file:
            game_type = int(re.search('<GO type="(\\d+)"', response.text).groups()[0])
            if (
                game_type & 0x10 or game_type & 0x04 or game_type & 0x02
            ):  # Filter out games with three players, no red tiles, or no riichi declaration
                return
            out_file.write(response.text)
    except Exception:
        return


def fetch_logs(filename: str) -> None:
    try:
        url = f"https://tenhou.net/sc/raw/dat/{filename}.html.gz"
        r = requests.get(url, headers=headers)
        r = gzip.decompress(r.content).decode("utf-8")
        with open(os.path.join(log_dir, f"{filename}.txt"), "w") as f:
            f.write(r)
    except Exception:
        return


def init_worker() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def download_logs() -> None:
    r = requests.get("https://tenhou.net/sc/raw/list.cgi", headers=headers)
    filenames = re.findall("scc\\d+.html.gz", r.text)
    exists = list(map(lambda x: x[:-4], os.listdir(log_dir)))
    filenames = set(map(lambda x: x[:-8], filenames))
    filenames.difference_update(exists)
    print(f"Found {len(filenames)} logs")
    with Pool(initializer=init_worker) as pool:
        list(tqdm.tqdm(pool.imap(fetch_logs, filenames), total=len(filenames)))


def download_data() -> None:
    links = []
    for log in os.listdir(log_dir):
        if log.endswith(".txt"):  # ignores gitkeep file
            with open(os.path.join(log_dir, log), "r") as f:
                data = f.readlines()
            for line in data:
                if "四鳳" in line:
                    link = (
                        re.search('<a href="(.*)">', line)
                        .groups()[0]
                        .replace("?log=", "log/?")
                    )
                    links.append(link)
    links = set(links)
    exists = list(
        map(
            lambda x: "http://tenhou.net/0/log/?" + x.split(".")[0],
            os.listdir(data_dir),
        )
    )
    links.difference_update(exists)
    print(f"Found {len(links)} links")
    with Pool(initializer=init_worker) as pool:
        list(tqdm.tqdm(pool.imap(fetch_data, links), total=len(links)))

def read_scc():
    # Iterate over all files in the directory
    for file in os.listdir(scc_dir):
        # Only process files that match the pattern 'scc*.html.gz'
        if file.startswith("scc") and file.endswith(".html.gz"):
            # Construct full file path
            full_file_path = os.path.join(scc_dir, file)
            # Construct new file path (remove .gz)
            new_file_path = full_file_path[:-3]
            # Unzip the file
            with gzip.open(full_file_path, "rb") as f_in:
                with open(new_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # Remove the original file
            os.remove(full_file_path)
            # Rename the file (change .html to .txt)
            os.rename(new_file_path, new_file_path.replace(".html", ".txt"))

    # Iterate over all files in the directory again
    for file in os.listdir(scc_dir):
        # Only process files that do not start with 'scc'
        if not file.startswith("scc") and not file.startswith(".git"):
            # Construct full file path
            full_file_path = os.path.join(scc_dir, file)
            # Remove the file
            os.remove(full_file_path)
