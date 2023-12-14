from pydantic import BaseModel


class ItemImgReqDto(BaseModel):
    itemImgUrl: str