from typing import List

from pydantic import BaseModel, Field


class Source(BaseModel):
    """schema for a source used by agent"""

    url: str = Field(description="The URL of the Source")


class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answerto the query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )
