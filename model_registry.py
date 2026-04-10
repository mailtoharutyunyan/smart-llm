"""
Component 18b: Model Registry — Version tracking, rollback, provenance.
Keeps last 3 adapters. Base model never modified.
"""
import json
import os
import shutil
import logging
import datetime
from typing import Optional

logger = logging.getLogger("acs.model_registry")


class ModelRegistry:
    """Tracks model versions with full provenance and rollback."""

    MAX_ADAPTERS = 3

    def __init__(
        self,
        registry_file: str = "./acs/models/registry.json",
        adapters_dir: str = "./acs/models/adapters/",
    ):
        self.registry_file = registry_file
        self.adapters_dir = adapters_dir
        self.versions: list[dict] = []
        self.current_version: str = "v1_base"
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        os.makedirs(self.adapters_dir, exist_ok=True)
        self._load()

    def _load(self):
        """Load registry from disk."""
        if os.path.exists(self.registry_file):
            with open(self.registry_file, "r") as f:
                data = json.load(f)
                self.versions = data.get("versions", [])
                self.current_version = data.get(
                    "current", "v1_base"
                )
        else:
            self.versions = [
                {
                    "version": "v1_base",
                    "tier1": 0.0,
                    "tier2": 0.0,
                    "created": datetime.datetime.now().isoformat(),
                    "dataset_version": None,
                    "status": "active",
                }
            ]
            self.current_version = "v1_base"
            self._save()

    def _save(self):
        """Persist registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(
                {
                    "current": self.current_version,
                    "versions": self.versions,
                },
                f,
                indent=2,
            )

    def register_new_model(
        self,
        dataset_version: Optional[str] = None,
        tier1: float = 0.0,
        tier2: float = 0.0,
    ) -> str:
        """Register a new model version."""
        new_v = f"v{len(self.versions) + 1}"
        entry = {
            "version": new_v,
            "tier1": tier1,
            "tier2": tier2,
            "created": datetime.datetime.now().isoformat(),
            "dataset_version": dataset_version,
            "status": "pending_eval",
        }
        self.versions.append(entry)
        self._save()
        logger.info("Registered new model: %s", new_v)
        return new_v

    def promote(self, version: str, tier1: float, tier2: float):
        """Mark a version as active after passing evaluation."""
        for v in self.versions:
            if v["version"] == version:
                v["tier1"] = tier1
                v["tier2"] = tier2
                v["status"] = "active"
                self.current_version = version
                break

        # Mark previous as archived
        for v in self.versions:
            if v["version"] != version and v["status"] == "active":
                v["status"] = "archived"

        self._prune_old_adapters()
        self._save()
        logger.info("Promoted %s to active (T1=%.3f, T2=%.3f)",
                    version, tier1, tier2)

    def reject(self, version: str, reason: str):
        """Mark a version as rejected."""
        for v in self.versions:
            if v["version"] == version:
                v["status"] = "rejected"
                v["rejection_reason"] = reason
                break
        self._save()
        logger.info("Rejected %s: %s", version, reason)

    def rollback(self, target_version: str) -> bool:
        """Rollback to a previous version."""
        found = False
        for v in self.versions:
            if v["version"] == target_version:
                if v["status"] in ("active", "archived"):
                    v["status"] = "active"
                    self.current_version = target_version
                    found = True
                else:
                    logger.error(
                        "Cannot rollback to %s (status: %s)",
                        target_version,
                        v["status"],
                    )
                    return False

        if found:
            for v in self.versions:
                if (
                    v["version"] != target_version
                    and v["status"] == "active"
                ):
                    v["status"] = "archived"
            self._save()
            logger.info("Rolled back to %s", target_version)
        return found

    def _prune_old_adapters(self):
        """Keep only last N adapters on disk."""
        adapter_versions = sorted(
            [
                v["version"]
                for v in self.versions
                if v["status"] in ("active", "archived")
                and v["version"] != "v1_base"
            ]
        )
        to_remove = adapter_versions[: -self.MAX_ADAPTERS]
        for ver in to_remove:
            path = os.path.join(self.adapters_dir, ver)
            if os.path.exists(path):
                shutil.rmtree(path)
                logger.info("Pruned old adapter: %s", ver)

    def get_current(self) -> dict:
        """Get details of the currently active model."""
        for v in self.versions:
            if v["version"] == self.current_version:
                return v
        return {"version": self.current_version, "status": "unknown"}

    def list_versions(self) -> list[dict]:
        """Return all registered versions."""
        return self.versions
