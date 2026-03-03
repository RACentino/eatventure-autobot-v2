import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.logger import setup_logger

logger = setup_logger("vision.scanner")


class AssetScanner:
    def __init__(self, image_matcher, max_workers=None):
        self.image_matcher = image_matcher
        cpu_count = os.cpu_count() or 1
        self.max_workers = max_workers or min(32, cpu_count + 4)
        self._template_cache = {}
        self._asset_index_cache = {}
        self._cache_lock = threading.RLock()

    def scan(self, assets_dir, required_templates=None):
        assets_path = Path(assets_dir)
        if not assets_path.exists():
            logger.error(f"Assets directory not found: {assets_path}")
            return {}

        required_set = set(required_templates or [])
        template_files = self._collect_template_files(assets_path, required_set)

        templates = {}
        if not template_files:
            return templates

        if len(template_files) == 1:
            template_name, template_data = self._load_template(template_files[0])
            if template_data is not None:
                templates[template_name] = template_data
                logger.info(f"Loaded template: {template_name}")
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._load_template, template_file): template_file
                    for template_file in template_files
                }
                for future in as_completed(futures):
                    template_file = futures[future]
                    try:
                        template_name, template_data = future.result()
                    except Exception as exc:
                        logger.error(f"Failed to load template {template_file}: {exc}")
                        continue

                    if template_data is None:
                        continue

                    templates[template_name] = template_data
                    logger.info(f"Loaded template: {template_name}")

        if required_set:
            missing = sorted(required_set.difference(templates.keys()))
            if missing:
                logger.warning(f"Missing {len(missing)} required templates: {', '.join(missing)}")

        return templates

    def _normalize_key(self, name):
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    def _collect_template_files(self, assets_path, required_set):
        indexed = self._index_assets_dir(assets_path)
        if required_set:
            template_files = []
            for template_name in required_set:
                indexed_path = indexed.get(template_name.lower())
                if indexed_path is None:
                    indexed_path = indexed.get(self._normalize_key(template_name))
                if indexed_path is not None:
                    template_files.append(indexed_path)
        else:
            template_files = list(indexed.values())
        template_files = sorted(set(template_files), key=lambda path: str(path).lower())
        return template_files

    def _index_assets_dir(self, assets_path):
        assets_key = str(assets_path)
        try:
            mtime = assets_path.stat().st_mtime
        except OSError:
            mtime = None

        with self._cache_lock:
            cached = self._asset_index_cache.get(assets_key)
            if cached and cached["mtime"] == mtime:
                return cached["index"]

        indexed = {}
        for template_path in assets_path.rglob("*"):
            if not template_path.is_file() or template_path.suffix.lower() != ".png":
                continue
            stem = template_path.stem
            indexed.setdefault(stem.lower(), template_path)
            indexed.setdefault(self._normalize_key(stem), template_path)

        with self._cache_lock:
            self._asset_index_cache[assets_key] = {"mtime": mtime, "index": indexed}
        return indexed

    def _load_template(self, template_file):
        template_name = template_file.stem
        try:
            mtime = template_file.stat().st_mtime
        except OSError:
            mtime = None

        key = str(template_file)
        with self._cache_lock:
            cached = self._template_cache.get(key)
            if cached and cached["mtime"] == mtime:
                return template_name, cached["data"]

        template_img = self.image_matcher.load_template(template_file)
        with self._cache_lock:
            self._template_cache[key] = {"mtime": mtime, "data": template_img}
        return template_name, template_img
