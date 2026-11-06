from typing import Optional

from pydantic import BaseModel, Field, field_validator


class SpecInput(BaseModel):
    """
    Validation model for SpecEngine execute input.
    """
    requirement_text: str = Field(..., description="The requirement description for the spec project")
    task_id: Optional[str] = Field(None, description="Optional task ID")

    @field_validator('requirement_text')
    def validate_requirement_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Requirement text cannot be empty")
        if len(v) > 50000:  # Reasonable upper limit
            raise ValueError("Requirement text is too long (max 50000 chars)")
        return v
