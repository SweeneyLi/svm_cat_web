from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User

admin.site.unregister(User)


@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    fieldsets = [
        (None, {'fields': ['username', 'password', 'email']}),
        ('Important dates', {'fields': ['last_login', 'date_joined'], 'classes': ['collapse']}),
    ]

    # list_display = ('question_text', 'pub_date', 'was_published_recently')
    # list_filter = ['pub_date']
    # search_fields = ['question_text']

