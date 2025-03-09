# FastAPI와 필요한 모듈들을 임포트
from typing import Union
import os
import mediapipe as mp  # 이미지 처리를 위한 MediaPipe 라이브러리
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, File, UploadFile   # FastAPI의 파일 업로드 처리를 위한 모듈
from fastapi.responses import JSONResponse

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np
from fastapi.responses import FileResponse


# MediaPipe 모델 설정 - 전역변수로 빼 놓는 것이 좋음
base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
classifier = vision.ImageClassifier.create_from_options(options)

base_options = python.BaseOptions(model_asset_path='models\efficientdet_lite0.tflite') 
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)


MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def visualize(
    image,
    detection_result
) -> np.ndarray:
    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x,
                        MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image



# FastAPI 인스턴스 생성
app = FastAPI()


# 루트 경로("/")에 대한 GET 요청을 처리하는 엔드포인트
# deep러닝하는데 대부분 python으로되어있어서  플라스크 웹사이트 개발용으로도 많이 만듬
@app.get("/")
def read_root():
    """
    루트 경로에 대한 기본 응답을 반환하는 함수
    Returns:
        dict: "Hello": "World"를 포함하는 JSON 응답
    """
    return {"Hello": "World"}



@app.get("/items/{item_id}")
def read_item(item_id: int = 1, q: Union[str, None] = "test"):
    return {"item_id": item_id, "q": q}



@app.post("/img_cls")
async def img_cls(
    image: UploadFile = File(...)  # '...'는 필수 파라미터를 의미

):
    # 이미지 파일 저장
    contents = await image.read()
    filename = f"temp_{image.filename}"
    with open(filename, "wb") as f:
        f.write(contents)
    
    # MediaPipe 모델 설정
    base_options = python.BaseOptions(model_asset_path='models/efficientnet_lite0.tflite')
    options = vision.ImageClassifierOptions(base_options=base_options, max_results=3)
    classifier = vision.ImageClassifier.create_from_options(options)

    # 이미지 로드 및 분류
    mp_image = mp.Image.create_from_file(filename)
    classification_result = classifier.classify(mp_image)

    # 결과 추출
    top_category = classification_result.classifications[0].categories[0]
    print(f"{top_category.category_name} ({top_category.score:.2f})")

    # 임시 파일 삭제
    os.remove(filename)
    
    return JSONResponse(
        content={
            "message": "이미지 분류 요청이 성공적으로 처리되었습니다",
            "image_filename": image.filename,
            "top_category": top_category.category_name,
            "score": float(top_category.score)  # float32를 JSON 직렬화 가능한 형태로 변환
        },
        status_code=200
    )


@app.post("/face_recognition")
async def face_recognition(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...)
):
    """
    얼굴 인식 API
    
    두 개의 이미지를 업로드하여 얼굴 인식을 수행합니다.
    
    - **image1**: 첫 번째 이미지 파일
    - **image2**: 두 번째 이미지 파일
    
    Returns:
        JSON 응답
    """
    # 여기에 얼굴 인식 로직을 구현할 수 있습니다
    pass
    
    return JSONResponse(
        content={
            "message": "얼굴 인식 요청이 성공적으로 처리되었습니다",
            "image1_filename": image1.filename,
            "image2_filename": image2.filename,
        },
        status_code=200
    )




@app.post("/img_det")
async def img_det(
    image: UploadFile = File(...)
):
    contents = await image.read()
    filename = f"temp_{image.filename}"
    with open(filename, "wb") as f:
        f.write(contents)
    
    # STEP 3: Load the input image.
    image = mp.Image.create_from_file(filename)
    # STEP 4: Detect objects in the input image.
    detection_result = detector.detect(image)
    # print(detection_result)
    # STEP 5: Process the detection result. In this case, visualize it.
    objects = []
    for detection in detection_result.detections:
        objects.append(detection)
    print(f"Find Object : {len(objects)}")
        
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite("test.jpg", rgb_annotated_image)
    os.remove(filename)
    return FileResponse("test.jpg")