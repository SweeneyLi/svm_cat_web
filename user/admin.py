from django.contrib import admin
from django.contrib.auth.models import User
from django.contrib.auth.admin import UserAdmin
from .models import UserProfile

admin.site.unregister(User)


class UserProfileInline(admin.StackedInline):
    model = UserProfile


@admin.register(User)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = [('id'), ('username'), ('last_login'), ('is_superuser')]

    search_fields = [('username')]

    fieldsets = [
        (None, {'fields': ['username', 'password', 'email']}),
        ('Important dates', {'fields': ['last_login', 'date_joined'], 'classes': ['collapse']}),
    ]

    inlines = [UserProfileInline]
