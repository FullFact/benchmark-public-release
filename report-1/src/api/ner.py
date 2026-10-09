"""
A module for finding named entities in text.
Uses the GliNER library for doing the entity recognition.
https://urchade.github.io/GLiNER/intro.html
"""

from __future__ import annotations

import logging
import threading
import warnings
from typing import TYPE_CHECKING, Any

from huggingface_hub.utils import disable_progress_bars

from api.models import Entity, EntityType

if TYPE_CHECKING:
    from gliner import GLiNER

ENTITY_TYPES: list[str] = list(EntityType)

# The GliNER model used for entity recognition. Change to a different pretrained
# model (e.g. gliner_small/gliner_large) to trade off accuracy against speed.
MODEL_NAME = "urchade/gliner_medium-v2.1"

_model: GLiNER | None = None
_model_lock = threading.Lock()


def _get_ner_model() -> GLiNER:
    global _model
    if _model is not None:
        return _model

    # Serialise first-time loading so concurrent callers (e.g. simultaneous API
    # requests) don't each download and instantiate the model. Double-checked:
    # the fast path above avoids taking the lock once the model is loaded.
    with _model_lock:
        if _model is not None:
            return _model

        # Imported lazily so that merely importing this module (or api.models)
        # does not pull in gliner/torch/transformers until NER is actually used.
        from gliner import GLiNER

        disable_progress_bars()
        # Silence Hugging Face Hub noise (deprecation warning + unauthenticated
        # request notice) without touching warning/log config for other modules.
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module="huggingface_hub.*")
            _model = GLiNER.from_pretrained(MODEL_NAME)
    return _model


def _to_entities(ents_raw: list[dict[str, Any]]) -> list[Entity]:
    return [
        Entity(
            text=ent_dict["text"],
            ent_type=ent_dict["label"],
            start=ent_dict["start"],
            end=ent_dict["end"],
            score=ent_dict["score"],
        )
        for ent_dict in ents_raw
    ]


def get_entities(text: str) -> list[Entity]:
    """Find named entities in a single text."""
    return get_entities_batch([text])[0]


def get_entities_batch(texts: list[str]) -> list[list[Entity]]:
    """Find named entities in many texts in one batched model pass.

    Returns one entity list per input text, in the same order. Preferred over
    calling get_entities in a loop, as a single pass amortises the per-call
    tokenisation and forward-pass overhead across the whole batch."""
    if not texts:
        return []
    model = _get_ner_model()
    batch_raw: list[list[dict[str, Any]]] = model.inference(texts, ENTITY_TYPES)
    return [_to_entities(ents_raw) for ents_raw in batch_raw]
