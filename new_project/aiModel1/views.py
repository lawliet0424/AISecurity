from django.shortcuts import render
from PIL import Image
from protectphoto.models import Photo

# Create your views here.
def process_image(input_image_path, output_image_path):
    photos = Photo.objects.all()  # UploadedPhoto 대신 Photo 사용

    for photo in photos:
        input_image_path = photo.image.path

        # Load the image
        input_image = Image.open(input_image_path)

        # Process the image with your OCR model
        processed_text = process_image_with_ocr(input_image)

        # Save the OCR result to the database
        photo.processed_text = processed_text
        photo.save()
    
    return redirect('result_list')

def result_list(request):
    # OCRResult 모델에서 결과 가져오기
    ocr_results = OCRResult.objects.all()

    return render(request, 'result_list.html', {'ocr_results': ocr_results})
