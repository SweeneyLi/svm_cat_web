# Generated by Django 2.1.5 on 2019-02-07 11:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('algorithm', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ModelTrainLog',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.IntegerField(verbose_name='user_id')),
                ('train_time', models.DateTimeField(auto_now_add=True, verbose_name='train_time')),
                ('model_name', models.CharField(max_length=100, verbose_name='model_name')),
                ('train_category_positive', models.CharField(max_length=100, verbose_name='train_category_positive')),
                ('positive_num', models.IntegerField(verbose_name='positive_num')),
                ('train_category_negative', models.CharField(max_length=100, verbose_name='train_category_negative')),
                ('negative_num', models.IntegerField(verbose_name='negative_num')),
                ('validation_size', models.IntegerField(verbose_name='validation_size')),
                ('accuracy_score', models.FloatField(default=0, max_length=100, verbose_name='accuracy_score')),
            ],
        ),
    ]
