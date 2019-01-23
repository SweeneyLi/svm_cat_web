from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import UserProfile

admin.site.unregister(User)


class UserProfileInline(admin.StackedInline):
    model = UserProfile


@admin.register(User)
class UserProfileAdmin(admin.ModelAdmin):
    fieldsets = [
        (None, {'fields': ['username', 'password']}),
        ('Important dates', {'fields': ['last_login', 'date_joined'], 'classes': ['collapse']}),
    ]
    inlines = [UserProfileInline]
    # list_display = ('question_text', 'pub_date', 'was_published_recently')
    # list_filter = ['pub_date']
    # search_fields = ['question_text']

