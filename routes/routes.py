from fastapi import APIRouter
from dto.commentReqDto import CommentReqDto
from dto.itemImgReqDto import ItemImgReqDto
from kobert import predict
from yolov8 import detect_color

api = APIRouter()

@api.post("/check-malicious-comment")
def checkMaliciousComment(commentReqDto : CommentReqDto):
    # TODO : Call cleanBot Metho
    return predict(commentReqDto.comment)

@api.post("/check-item-color")
def checkMaliciousComment(itemImgReqDto : ItemImgReqDto):
    # TODO : Call cleanBot Method
    return detect_color(itemImgReqDto.itemImgUrl)
