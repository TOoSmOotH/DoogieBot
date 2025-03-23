from typing import Optional

from pydantic import BaseModel


class Token(BaseModel):
    """
    Token schema for authentication.
    """

    access_token: str
    refresh_token: str
    token_type: str


class TokenPayload(BaseModel):
    """
    Token payload schema.
    """

    sub: Optional[str] = None
    exp: Optional[int] = None


class RefreshToken(BaseModel):
    """
    Refresh token schema.
    """

    refresh_token: str
