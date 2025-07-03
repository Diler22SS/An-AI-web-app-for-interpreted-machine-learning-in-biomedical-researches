# ADMIN.PY
from django.contrib import admin
from .models import (
    Pipeline, FS_Filter, FS_Wrapper, ML_Model
)


@admin.register(FS_Filter)
class FSFilterAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'name',
        'function_name',
        'adjustable_parameters',
        'description'
    )


@admin.register(FS_Wrapper)
class FS_WrapperAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'name',
        'function_name',
        'adjustable_parameters',
        'description'
    )


@admin.register(ML_Model)
class ML_ModelAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'name',
        'function_name',
        'adjustable_parameters',
        'description'
    )


@admin.register(Pipeline)
class PipelineAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        "filename",
        "target_variable",
        "preliminarily_selected_features",
        "fs_filter__name",
        "fs_filter_parameters",
        "fs_filter_selected_features",
        "fs_wrapper__name",
        "fs_wrapper_parameters",
        "fs_wrapper_selected_features",
        "final_selected_features",
        "ml_model__name",
        "ml_model_parameters",
        "ml_model_metrics",
        # "ml_model_shap_values",
    )
