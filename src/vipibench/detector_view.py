from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vipibench.episode import EpisodeLabel, ExecutableEpisode


class DetectorView(BaseModel):
    """A label-bearing evaluation row with a deliberately narrow model-visible surface."""

    model_config = ConfigDict(extra="forbid")

    episode_id: str
    label: EpisodeLabel
    trusted_goal_text: str = Field(min_length=1)
    retrieved_text: str = Field(min_length=1)

    @property
    def source_role(self) -> str:
        return "user+retrieved"

    @property
    def trust_level(self) -> str:
        return "trusted+untrusted"

    def model_input(
        self,
        mode: Literal["text_only", "role_only", "text_role_provenance"],
    ) -> str:
        segments = sorted(
            (
                (self.trusted_goal_text, "USER", "TRUSTED"),
                (self.retrieved_text, "RETRIEVED", "UNTRUSTED"),
            ),
            key=lambda item: hashlib.sha256(item[0].encode("utf-8")).hexdigest(),
        )
        if mode == "text_only":
            return " [SEGMENT] ".join(text for text, _, _ in segments)
        provenance = " [SEGMENT] ".join(
            sorted(
                [
                    "[ROLE=USER][TRUST=TRUSTED]",
                    "[ROLE=RETRIEVED][TRUST=UNTRUSTED]",
                ]
            )
        )
        if mode == "role_only":
            return provenance
        if mode == "text_role_provenance":
            return " [SEGMENT] ".join(
                f"[ROLE={role}][TRUST={trust}] {text}"
                for text, role, trust in segments
            )
        raise ValueError(f"unsupported detector input mode: {mode}")


def detector_view_from_episode(episode: ExecutableEpisode) -> DetectorView:
    trusted = [item for item in episode.context if item.chunk_id == "trusted-user"]
    retrieved = [item for item in episode.context if item.chunk_id == "retrieved-context"]
    if len(trusted) != 1 or len(retrieved) != 1:
        raise ValueError(f"expected one trusted and one retrieved chunk: {episode.episode_id}")
    trusted_chunk = trusted[0]
    retrieved_chunk = retrieved[0]
    if trusted_chunk.role.value != "user" or trusted_chunk.trust.value != "trusted":
        raise ValueError(f"trusted context boundary mismatch: {episode.episode_id}")
    if retrieved_chunk.role.value != "retrieved" or retrieved_chunk.trust.value != "untrusted":
        raise ValueError(f"detector context boundary mismatch: {episode.episode_id}")
    return DetectorView(
        episode_id=episode.episode_id,
        label=episode.label,
        trusted_goal_text=trusted_chunk.content,
        retrieved_text=retrieved_chunk.content,
    )
