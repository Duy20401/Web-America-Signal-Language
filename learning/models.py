from django.db import models
from django.urls import reverse

class Alphabet(models.Model):
    letter = models.CharField(max_length=1, unique=True, verbose_name="Chữ cái")
    image = models.ImageField(upload_to='alphabet_images/', verbose_name="Hình ảnh")
    description = models.TextField(blank=True, verbose_name="Mô tả")
    video_url = models.URLField(blank=True, verbose_name="URL video hướng dẫn")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Chữ cái"
        verbose_name_plural = "Bảng chữ cái"
        ordering = ['letter']
    
    def __str__(self):
        return f"Chữ cái {self.letter}"
    
    def get_absolute_url(self):
        return reverse('alphabet_detail', kwargs={'letter': self.letter})

class Word(models.Model):
    CATEGORY_CHOICES = [
        ('basic', 'Cơ bản'),
        ('greeting', 'Chào hỏi'),
        ('family', 'Gia đình'),
        ('food', 'Đồ ăn'),
    ]
    
    word = models.CharField(max_length=50, unique=True, verbose_name="Từ")
    category = models.CharField(max_length=20, choices=CATEGORY_CHOICES, default='basic')
    image = models.ImageField(upload_to='word_images/', verbose_name="Hình ảnh")
    description = models.TextField(blank=True, verbose_name="Mô tả")
    video_url = models.URLField(blank=True, verbose_name="URL video hướng dẫn")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Từ vựng"
        verbose_name_plural = "Từ vựng"
        ordering = ['word']
    
    def __str__(self):
        return f"Từ: {self.word}"
    
    def get_absolute_url(self):
        return reverse('word_detail', kwargs={'word': self.word})