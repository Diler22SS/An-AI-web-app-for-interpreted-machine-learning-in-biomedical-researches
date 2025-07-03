# FORMS.PY
from django import forms


class FileUploadForm(forms.Form):
    """Форма для валидации загруженных файлов набора данных."""

    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    dataset_file = forms.FileField(
        required=True,
        widget=forms.ClearableFileInput(
            attrs={'accept': '.csv,.xlsx,.xls'}
        ),
        help_text=f"Поддерживаемые форматы: {', '.join(ALLOWED_EXTENSIONS)}. Макс. размер: 10MB"
    )

    def clean_dataset_file(self):
        """Валидация загруженного набора данных."""
        file = self.cleaned_data['dataset_file']

        extension = file.name.split('.')[-1].lower() if '.' in file.name else ''
        if extension not in self.ALLOWED_EXTENSIONS:
            raise forms.ValidationError(
                f"Неподдерживаемый формат файла. Разрешены: {', '.join(self.ALLOWED_EXTENSIONS)}."
            )

        if file.size > self.MAX_FILE_SIZE:
            raise forms.ValidationError(
                "Файл слишком большой. Максимальный размер 10MB."
            )

        return file


class TargetVariableForm(forms.Form):
    """Форма для выбора целевой переменной."""

    target_variable = forms.ChoiceField(
        required=True,
        choices=(),
        widget=forms.Select(attrs={'class': 'form-select'}),
        label="Целевая переменная",
        help_text="Выберите целевую переменную"
    )

    def __init__(self, *args, features=None, **kwargs):
        """Инициализация формы."""

        super().__init__(*args, **kwargs)

        if features:
            choices = [('', 'Выберите целевую переменную')] + [(f, f) for f in features]
            self.fields['target_variable'].choices = choices


class FeatureSelectionForm(forms.Form):
    """Форма предварительного выбора признаков."""

    selected_features = forms.MultipleChoiceField(
        required=False,
        widget=forms.CheckboxSelectMultiple(
            attrs={'class': 'feature-checkbox'}
        )
    )

    def __init__(self, *args, features=None, target_variable=None, **kwargs):
        """Инициализация формы."""
        super().__init__(*args, **kwargs)

        if features and target_variable:
            available_features = [f for f in features if f != target_variable]
            choices = [(f, f) for f in available_features]
            self.fields['selected_features'].choices = choices

            self.all_features = features
            self.target_variable = target_variable
