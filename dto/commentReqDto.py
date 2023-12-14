from pydantic import BaseModel


class CommentReqDto(BaseModel):
    comment: str