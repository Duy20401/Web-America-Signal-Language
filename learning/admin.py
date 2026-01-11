from django.contrib import admin
from .models import Alphabet, Word

@admin.register(Alphabet)
class AlphabetAdmin(admin.ModelAdmin):
    list_display = ['letter', 'description']
    search_fields = ['letter']

@admin.register(Word)
class WordAdmin(admin.ModelAdmin):
    list_display = ['word', 'category', 'description']
    search_fields = ['word']
    list_filter = ['category']