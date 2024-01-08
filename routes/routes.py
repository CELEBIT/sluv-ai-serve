from fastapi import APIRouter
import os
import sys
sys.path.append(os.getcwd())
from dto.commentReqDto import CommentReqDto
from dto.itemImgReqDto import ItemImgReqDto
import kobert_model
import yolov8

api = APIRouter()

@api.post("/check-malicious-comment")
def checkMaliciousComment(commentReqDto : CommentReqDto):
    # TODO : Call cleanBot Metho
    return kobert_model.predict(commentReqDto.comment)

@api.post("/check-item-color")
def checkItemColor(itemImgReqDto : ItemImgReqDto):
    # TODO : Call cleanBot Method
    return "#" + yolov8.detect_color(itemImgReqDto.itemImgUrl)[0]
