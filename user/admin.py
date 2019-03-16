from django.contrib.auth.models import Group, User
from django.contrib import admin
from .models import UserProfile

admin.site.unregister(Group)

admin.site.unregister(User)


class UserProfileInline(admin.StackedInline):
    model = UserProfile


@admin.register(User)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = [('id'), ('username'), ('last_login'), ('is_superuser'), ('first_name'), ('last_name')]

    search_fields = [('username')]

    fieldsets = [
        (None, {'fields': ['username', 'password', 'email', 'first_name', 'last_name']}),
        ('Important dates', {'fields': ['last_login', 'date_joined'], 'classes': ['collapse']}),
    ]

    inlines = [UserProfileInline]
