from collections import deque
import hashlib
import os
import shutil
import subprocess
import time

from huggingface_hub import snapshot_download, scan_cache_dir

class WeightsDownloadCache:
    def __init__(
        self, min_disk_free: int = 1 * (2**30), base_dir: str = "/src/weights-cache"
    ):
        """
        WeightsDownloadCache is meant to track and download weights files as fast
        as possible, while ensuring there's enough disk space.

        It tries to keep the most recently used weights files in the cache, so
        ensure you call ensure() on the weights each time you use them.

        It will not re-download weights files that are already in the cache.

        :param min_disk_free: Minimum disk space required to start download, in bytes.
        :param base_dir: The base directory to store weights files.
        """
        self.min_disk_free = min_disk_free # default 1 GB
        self.base_dir = base_dir
        self._hits = 0
        self._misses = 0

        # Least Recently Used (LRU) cache for paths
        self.lru_paths = deque()
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def _remove_least_recent(self) -> None:
        """
        Remove the least recently used weights file from the cache and disk.
        """
        oldest = self.lru_paths.popleft()
        self._rm_disk(oldest)

    def cache_info(self) -> str:
        """
        Get cache information.

        :return: Cache information.
        """

        return f"CacheInfo(hits={self._hits}, misses={self._misses}, base_dir='{self.base_dir}', currsize={len(self.lru_paths)})"

    def _rm_disk(self, path: str) -> None:
        """
        Remove a weights file or directory from disk.
        :param path: Path to remove.
        """
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

    def _has_enough_space(self) -> bool:
        """
        Check if there's enough disk space.

        :return: True if there's more than min_disk_free free, False otherwise.
        """
        disk_usage = shutil.disk_usage(self.base_dir)
        print(f"Free disk space: {disk_usage.free}")
        return disk_usage.free >= self.min_disk_free

    def ensure(self, lora_id: str) -> str:
        """
        Ensure weights file is in the cache and return its path.

        This also updates the LRU cache to mark the weights as recently used.

        :param url: URL to download weights file from, if not in cache.
        :return: Path to weights.
        """
        path = self.weights_path(lora_id)

        if path in self.lru_paths and path is not None:
            # here we remove to re-add to the end of the LRU (marking it as recently used)
            self._hits += 1
            self.lru_paths.remove(path)
        else:
            self._misses += 1
            self.download_weights(lora_id)

        self.lru_paths.append(path)  # Add file to end of cache
        return path

    def weights_path(self, lora_id: str) -> str:
        """Get the path to store weights file from lora_id."""
        data = scan_cache_dir(self.base_dir)
        
        for repo in data.repos:
            if(repo.repo_id == lora_id):
                return str(repo.repo_path)
        
        return None

    def download_weights(self, model_id) -> None:
        """
        Download weights file from a URL, ensuring there's enough disk space.

        :param model_id: Id of the lora model galverse/mama-1.5
        """
        print("Ensuring enough disk space...")
        while not self._has_enough_space() and len(self.lru_paths) > 0:
            self._remove_least_recent()

        print(f"Downloading weights: {model_id}")

        st = time.time()
        
        if(self.check_model_exist_in_cache(model_id)):
            print(f"Model {model_id} already exists in cache. Skip downloading...")
            return
        
        snapshot_download(repo_id=model_id, cache_dir=self.base_dir)
        print(f"Downloaded weights in {time.time() - st} seconds")

    def check_model_exist_in_cache(self, model_id: str): 
        data = scan_cache_dir(self.base_dir)
        
        for repo in data.repos:
            if(repo.repo_id == model_id):
                True
        
        return None