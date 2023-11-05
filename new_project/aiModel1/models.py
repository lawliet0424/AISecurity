from django.db import models

class OCRResult(models.Model):
    image = models.ImageField(upload_to='processed_images/')  # OCR 처리된 이미지를 저장할 필드
    processed_text = models.TextField()  # OCR 처리 결과 텍스트를 저장할 필드
    timestamp = models.DateTimeField(auto_now_add=True)  # 저장 시간을 저장할 필드

    def __str__(self):
        return f"OCR Result for {self.timestamp}"

