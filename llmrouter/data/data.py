from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

class Data(BaseModel):
    pk: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_name: Optional[str] = Field(default=None)


class Profile(Data):
    name: str
    bio: str
    collaborators: Optional[List[str]] = Field(default=[])
    pub_titles: Optional[List[str]] = Field(default=[])
    pub_abstracts: Optional[List[str]] = Field(default=[])
    domain: Optional[List[str]] = Field(default=[])
    institute: Optional[str] = Field(default=None)
    embed: Optional[Any] = Field(default=None)
    is_leader_candidate: Optional[bool] = Field(default=True)
    is_member_candidate: Optional[bool] = Field(default=True)
    is_reviewer_candidate: Optional[bool] = Field(default=True)
    is_chair_candidate: Optional[bool] = Field(default=True)