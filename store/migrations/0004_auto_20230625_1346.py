# Generated by Django 3.2 on 2023-06-25 08:16

import datetime
from django.db import migrations, models
import django.db.models.deletion
import store.models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('store', '0003_product_expiry_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='unique_key',
            field=models.UUIDField(default=uuid.uuid4, editable=False, unique=True),
        ),
        migrations.AlterField(
            model_name='product',
            name='id',
            field=models.BigAutoField(primary_key=True, serialize=False),
        ),
        migrations.CreateModel(
            name='ExpiredMedecines',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('slug', models.CharField(max_length=150)),
                ('name', models.CharField(max_length=150)),
                ('product_image', models.ImageField(blank=True, null=True, upload_to=store.models.get_file_path)),
                ('small_description', models.CharField(max_length=250)),
                ('quantity', models.IntegerField()),
                ('description', models.TextField(max_length=500)),
                ('original_price', models.FloatField()),
                ('selling_price', models.FloatField()),
                ('tag', models.CharField(max_length=150)),
                ('meta_title', models.CharField(max_length=150)),
                ('meta_keywords', models.CharField(max_length=500)),
                ('meta_description', models.TextField(max_length=500)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('expiry_date', models.DateField(default=datetime.datetime.now)),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='store.product')),
            ],
        ),
    ]
