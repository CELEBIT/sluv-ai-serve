from fastapi import APIRouter
from dto.commentReqDto import CommentReqDto
from dto.itemImgReqDto import ItemImgReqDto

api = APIRouter()

@api.post("/check-malicious-comment")
def checkMaliciousComment(commentReqDto : CommentReqDto):
    # TODO : Call cleanBot Metho
    return "POST /check-malicious-comment. BODY:" + commentReqDto.comment

@api.post("/check-item-color")
def checkMaliciousComment(itemImgReqDto : ItemImgReqDto):
    # TODO : Call cleanBot Method
    return "POST /check-item-color. BODY:" + itemImgReqDto.itemImgUrl
