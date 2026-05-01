import os
from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator


class MossConfig(BaseModel):
    collection_name: str = Field("mem0", description="Moss index name used as the mem0 collection")
    project_id: str = Field(
        default_factory=lambda: os.getenv("MOSS_PROJECT_ID", ""),
        description="Moss project ID — falls back to MOSS_PROJECT_ID env var",
    )
    project_key: str = Field(
        default_factory=lambda: os.getenv("MOSS_PROJECT_KEY", ""),
        description="Moss project key — falls back to MOSS_PROJECT_KEY env var",
    )
    model_id: str = Field(
        "moss-minilm",
        description="Embedding model used when creating the index. 'moss-minilm' (fast) or 'moss-mediumlm' (more accurate)",
    )
    alpha: float = Field(
        0.8,
        description="Hybrid search weight: 1.0 = pure semantic, 0.0 = pure keyword, 0.8 = default",
        ge=0.0,
        le=1.0,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_credentials(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        project_id = values.get("project_id") or os.getenv("MOSS_PROJECT_ID", "")
        project_key = values.get("project_key") or os.getenv("MOSS_PROJECT_KEY", "")
        if not project_id:
            raise ValueError("project_id is required (or set MOSS_PROJECT_ID env var)")
        if not project_key:
            raise ValueError("project_key is required (or set MOSS_PROJECT_KEY env var)")
        return values

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed = set(cls.model_fields.keys())
        extra = set(values.keys()) - allowed
        if extra:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(sorted(extra))}. "
                f"Allowed fields: {', '.join(sorted(allowed))}"
            )
        return values
