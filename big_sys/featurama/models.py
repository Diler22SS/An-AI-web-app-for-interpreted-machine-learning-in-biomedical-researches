# MODELS.PY
from django.db import models
import pandas as pd
from io import StringIO


class FS_Filter(models.Model):
    """ FS_Filter """
    name = models.CharField(max_length=255, null=True, blank=True)
    function_name = models.CharField(max_length=255, null=True, blank=True)
    adjustable_parameters = models.JSONField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']
        verbose_name = "Фильрующий метод отбора признаков"
        verbose_name_plural = "Справочник фильрующих методов отбора признаков"


class FS_Wrapper(models.Model):
    """ FS_Wrapper """
    name = models.CharField(max_length=255, null=True, blank=True)
    function_name = models.CharField(max_length=255, null=True, blank=True)
    adjustable_parameters = models.JSONField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']
        verbose_name = "Метод-обертка отбора признаков"
        verbose_name_plural = "Справочник методов-оберток отбора признаков"


class ML_Model(models.Model):
    """ ML_Model """
    name = models.CharField(max_length=255, null=True, blank=True)
    function_name = models.CharField(max_length=255, null=True, blank=True)
    adjustable_parameters = models.JSONField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name

    class Meta:
        ordering = ['name']
        verbose_name = "Модель машинного обучения"
        verbose_name_plural = "Справочник моделей машинного обучения"


class Pipeline(models.Model):
    """ Pipeline """
    # Dataset
    filename = models.CharField(max_length=255, null=True, blank=True)
    data_content = models.JSONField(null=True, blank=True)
    target_variable = models.CharField(max_length=255, null=True, blank=True)
    preliminarily_selected_features = models.JSONField(null=True, blank=True)

    # FS_Filter
    fs_filter = models.ForeignKey(
        FS_Filter,
        on_delete=models.PROTECT,
        related_name='pipelines',
        null=True,
        blank=True
    )
    fs_filter_parameters = models.JSONField(null=True, blank=True)
    fs_filter_selected_features = models.JSONField(null=True, blank=True)

    # FS_Wrapper
    fs_wrapper = models.ForeignKey(
        FS_Wrapper,
        on_delete=models.PROTECT,
        related_name='pipelines',
        null=True,
        blank=True
    )
    fs_wrapper_parameters = models.JSONField(null=True, blank=True)
    fs_wrapper_selected_features = models.JSONField(null=True, blank=True)
    final_selected_features = models.JSONField(null=True, blank=True)

    # ML_Model
    ml_model = models.ForeignKey(
        ML_Model,
        on_delete=models.PROTECT,
        related_name='pipelines',
        null=True,
        blank=True
    )
    ml_model_parameters = models.JSONField(null=True, blank=True)
    ml_model_metrics = models.JSONField(null=True, blank=True)
    ml_model_shap_values = models.JSONField(null=True, blank=True)

    def __str__(self):
        return f"Пайплайн №{self.pk} для набора данных \"{self.filename}\""

    def get_dataframe(self) -> pd.DataFrame:
        """Return the data as a pandas DataFrame."""
        if not self.data_content:
            return pd.DataFrame()
        return pd.read_json(StringIO(self.data_content))

    def save_dataframe(self, df: pd.DataFrame) -> None:
        """Save a pandas DataFrame as JSON."""
        self.data_content = df.to_json(orient='records')
        self.save()

    class Meta:
        ordering = ['-id']
        verbose_name = "Пайплайн"
        verbose_name_plural = "Пайплайны"
