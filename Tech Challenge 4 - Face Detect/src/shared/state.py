from __future__ import annotations

from dataclasses import dataclass, field

from shared.config import Settings
from shared.models import AlertItem, MultimodalEvent


@dataclass
class AppState:
    settings: Settings
    events: list[MultimodalEvent] = field(default_factory=list)
    alerts: list[AlertItem] = field(default_factory=list)
    event_details: dict[str, dict] = field(default_factory=dict)
