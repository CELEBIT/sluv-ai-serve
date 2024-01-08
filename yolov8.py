from ultralytics import YOLO
from PIL import Image
import numpy as np
import requests
from io import BytesIO

model = YOLO("/home/test/yolo.pt")

# 이미지 받는 경로(url)
url = 'https://www.muji.com/wp-content/uploads/sites/12/2021/02/015.jpg'

def detect_color(path):
    color_result = []
    res = requests.get(path).content
    img = Image.open(BytesIO(res))
    pix = np.array(img)
    results = model(img)
    a = results[0].boxes.xywh.detach().cpu().squeeze().numpy().reshape(-1,4)
    
    for i in a:
        rgb = pix[int(i[1])][int(i[0])]
        hexa = ''

        for ele in rgb:
            if ele < 16:
                hexa_answer = '0' + "{0:x}".format(ele)
                hexa += hexa_answer
            else:
                hexa_answer = "{0:x}".format(ele)
                hexa += hexa_answer
        
        color_result.append(hexa)

    return color_result

print(detect_color(url))
