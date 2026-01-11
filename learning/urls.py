from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('alphabet/', views.learn_alphabet, name='learn_alphabet'),
    path('alphabet/letters/', views.learn_letters, name='learn_letters'),
    path('alphabet/digits/', views.learn_digits, name='learn_digits'),
    path('words/', views.practice_words_v2, name='learn_words'),
    path('words/<str:word>/', views.word_detail, name='word_detail'),
    path('practice/', views.practice, name='practice'),
    path('practice/camera/', views.practice_camera, name='practice_camera'),
    path('practice/words/', views.practice_words_camera, name='practice_words_camera'),
    
    # API for real-time recognition
    path('api/recognize/', views.api_recognize, name='api_recognize'),
    path('api/recognize/words/', views.api_recognize_words_v2, name='api_recognize_words'),
    path('api/letters/', views.api_letters_signed_urls, name='api_letters_signed_urls'),
    path('api/digits/', views.api_digits, name='api_digits'),
    path('api/words/', views.api_words_list, name='api_words_list'),
    path('api/vocabulary/', views.api_vocabulary_items, name='api_vocabulary_items'),
    
    # Must be last - catches any remaining alphabet/<letter>/
    path('alphabet/<str:letter>/', views.alphabet_detail, name='alphabet_detail'),

]