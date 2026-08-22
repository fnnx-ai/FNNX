# ==============================================================
# This file was automatically copied from spec.
# DO NOT EDIT — changes here will be overwritten.
# ==============================================================

from pydantic import BaseModel, ConfigDict


class MetaEntry(BaseModel):
    # Unknown top-level keys are preserved rather than dropped, so an entry round-trips.
    model_config = ConfigDict(extra="allow")

    id: str
    producer: str
    producer_version: str
    producer_tags: list[str]
    payload: dict
