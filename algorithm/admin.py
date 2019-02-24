from django.contrib import admin
from .models import SVMModel, ModelTrainLog


@admin.register(SVMModel)
class SVMModelAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_id', 'model_name', 'comment', 'train_num', 'update_time', 'recently_accuracy_score')

    search_fields = ('user_id', 'model_name')

    list_filter = ['user_id']


@admin.register(ModelTrainLog)
class ModelTrainLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'user_id', 'train_time', 'model_id', 'accuracy_score')

    search_fields = ('user_id', 'model_id')

    list_filter = ['user_id', 'model_id']
