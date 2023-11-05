from django.http import HttpResponseRedirect
from django.urls import reverse
from django.http import HttpResponse
from django.shortcuts import redirect
from django.shortcuts import get_object_or_404
from django.shortcuts import render
from .forms import PhotoForm
from .models import Photo
from .models import OCRResult
import os
import numpy as np
import platform
from matplotlib import pyplot as pyplot
import matplotlib.pyplot as plt
from .models import OCRResult
from django.conf import settings


import uuid
import json
import time
import cv2
import requests
import os
import csv
import re

api_url = 'https://qsf54xe2bm.apigw.ntruss.com/custom/v1/24922/752248f4ef47186d702b37d4e586420160ac6cf3e46ac1fe4d56b181017495d9/general'
secret_key = 'QUF0alJ4a3VrRWZzRUZUYWJKRlVrVHlVd0l2bXlUS3o='

def read_text_to_blur_from_csv(csv_file):
    text_to_blur = []
    with open(csv_file, 'r', newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row:
                words = row[0].split()  # 단어 단위로 분할
                text_to_blur.extend(words)  # 단어를 리스트에 추가
    return text_to_blur

    # CSV 파일에서 블러링을 적용할 글자 목록 읽기

# 이미지를 처리하고 저장하는 함수 정의
def process_and_save_image(image_path, text_to_blur, output_folder):
    # 이미지 파일이 존재하는지 확인
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        return

    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
    roi_img = img.copy()

    # CLOVA OCR API에 요청 보내기
    image_file = ('file', open(image_path, 'rb'))  # 이미지 파일을 튜플로 래핑

    request_json = {'images': [{'format': 'jpg', 'name': 'demo'}],
                    'requestId': str(uuid.uuid4()),
                    'version': 'V2',
                    'timestamp': int(round(time.time() * 1000))
                    }

    payload = {'message': json.dumps(request_json).encode('UTF-8')}

    headers = {
        'X-OCR-SECRET': secret_key,
    }

    response = requests.request("POST", api_url, headers=headers, data=payload, files=[image_file])  # 이미지 파일을 리스트로 전달
    result = response.json()

    # OCR 결과 추출
    def extract_text(result):
        text = ""
        for field in result['images'][0]['fields']:
            text += field['inferText'] + "\n"
        return text.strip()

    text = extract_text(result)

    # 텍스트 후처리 함수
    def post_process_text(text):
        # 불필요한 공백 및 특수 문자 제거
        text = re.sub(r"[^a-zA-Z0-9가-힣\s]", "", text)

        # 문자열 정규화 (대소문자 통일, 띄어쓰기 교정 등)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)  # 중복 공백 제거

        # OCR에서 발생한 일반적인 오류 수정
        # (예를 들어, 'O'와 '0'을 구분하기 위한 교정)
        text = text.replace('o', '0')

        return text

    # 텍스트 후처리
    processed_text = post_process_text(text)

    # 추출된 텍스트 라인을 저장할 리스트
    lines = []

    # OCR 결과 처리 및 블러링 적용
    for field in result['images'][0]['fields']:
        text = field['inferText']
        vertices_list = field['boundingPoly']['vertices']
        pts = [tuple(vertice.values()) for vertice in vertices_list]
        x_coords = [int(_) for _ in [pts[0][0], pts[1][0], pts[2][0], pts[3][0]]
                    ]
        y_coords = [int(_) for _ in [pts[0][1], pts[1][1], pts[2][1], pts[3][1]]
                    ]

        # 텍스트 라인을 (x, y) 좌표와 텍스트로 저장
        lines.append({'x_coords': x_coords, 'y_coords': y_coords, 'text': text})

    # 텍스트 라인을 y 좌표를 기준으로 정렬
    lines.sort(key=lambda line: np.mean(line['y_coords']))

    # 블러링 적용
    for line in lines:
        text = line['text']
        if any(word in text for word in text_to_blur) or text.endswith("ro") or text.endswith("gil") or text.endswith(
                "로") or text.endswith("길"):  # 블러링을 적용해야 할 텍스트인 경우
            x_min = min(line['x_coords'])
            x_max = max(line['x_coords'])
            y_min = min(line['y_coords'])
            y_max = max(line['y_coords'])

            if x_max > x_min and y_max > y_min:
                if roi_img is not None:
                    roi = img[y_min:y_max, x_min:x_max]
                # 이미지가 비어 있지 않은 경우에만 블러링을 적용
                if roi is not None and not roi.size == 0:
                    roi = cv2.GaussianBlur(roi, (0, 0), sigmaX=10)  # 블러링 효과 적용
                    img[y_min:y_max, x_min:x_max] = roi

    # 이미지 저장
    cv2.imwrite(output_path, img)

    # OCR 결과를 모델 필드에 할당
    ocr_result = OCRResult()
    ocr_result.image_path = output_path  # 이미지 경로를 모델 필드에 할당
    ocr_result.processed_text = processed_text  # OCR 결과를 모델 필드에 할당
    ocr_result.save()

    return ocr_result

def upload_file(request):
    ocr_results = []
    if request.method == 'POST':
        form = PhotoForm(request.POST, request.FILES)
        if form.is_valid():
            
            for file in request.FILES.getlist('image'):
                # 이미지를 저장
                photo_instance = Photo.objects.create(image=file)

                # OCR 처리 (aiModel1 앱의 process_and_save_image 함수를 사용)
                image_path = settings.MEDIA_ROOT + str(photo_instance.image)
                output_folder = os.path.join(settings.MEDIA_ROOT, 'output_folder')  # OCR 처리 결과를 저장할 폴더 경로

                # aiModel1 앱의 ocr_three.py 내의 함수 호출
                csv_file = os.path.join(settings.MEDIA_ROOT, 'Total_Road_Address.csv')
                text_to_blur = read_text_to_blur_from_csv(csv_file)
                result = process_and_save_image(image_path, text_to_blur, output_folder)
                ocr_results.append(result)

            return render(request, 'result_list.html', {'ocr_results': ocr_results})
    else:
        form = PhotoForm()
    return render(request, 'upload.html', {'form': form})

def image_list(request):
    photos = Photo.objects.all()
    return render(request, 'image_list.html', {'photos': photos})

def download_and_delete_image(request):

     # 모든 이미지를 가져옵니다. 이 예제에서는 모든 이미지를 가져오지만 필요에 따라 필터링할 수 있습니다.
    all_images = Photo.objects.all()

    # 다운로드 및 삭제된 이미지 수를 추적하는 변수를 초기화합니다.
    downloaded_images = 0

    for photo_instance in all_images:
        # 이미지 파일 경로를 가져옵니다.
        image_path = photo_instance.image.path

        with open(image_path, 'rb') as image_file:
            response = HttpResponse(image_file.read(), content_type='image/jpeg')
            response['Content-Disposition'] = f'attachment; filename="{photo_instance.image.name}"'

        if os.path.exists(image_path):
            # 이미지 파일 삭제
            os.remove(image_path)

            # 이미지 모델 삭제
            photo_instance.delete()

            downloaded_images += 1

    # 모든 이미지를 처리한 후 응답을 반환합니다.
    return HttpResponse(f'Downloaded and deleted {downloaded_images} images.')



def result_list(request):
    # OCRResult 모델에서 결과 가져오기
    ocr_results = OCRResult.objects.all()

    return render(request, 'result_list.html', {'ocr_results': ocr_results})