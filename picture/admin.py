from django.contrib import admin
from .models import Picture


@admin.register(Picture)
class PictureAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_id', 'pic_name', 'path', 'category', 'pic_size', 'upload_date')

    search_fields = ('user_id', 'pic_name')

    list_filter = ['user_id']
