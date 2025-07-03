# UTILS.PY
from typing import Tuple, Optional, List
import pandas as pd
import importlib
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def read_dataset_file(file) -> Tuple[pd.DataFrame, str]:
    """Read a dataset file and return a dataframe."""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        df = pd.read_excel(file, engine='openpyxl')
    elif file.name.endswith('.xls'):
        df = pd.read_excel(file, engine='xlrd')
    else:
        # по умолчанию CSV
        df = pd.read_csv(file)
    return df


def validate_dataset(
    df: pd.DataFrame,
    target_variable: str,
    selected_features: List[str]
) -> Tuple[bool, Optional[str]]:
    """Валидация набора данных на соотвествие требованиям."""
    # Проверка, что выбранные признаки существуют в наборе данных
    missing = set(selected_features) - set(df.columns)
    if missing:
        msg = f"Отсутствующие колонки: {missing}"
        return False, msg

    # Проверка выбранных данных
    columns_to_check = selected_features + [target_variable]
    df_subset = df[columns_to_check]

    # 1. Проверка на пропущенные значения
    missing_check, missing_error = check_no_missing_values(df_subset)
    if not missing_check:
        return False, missing_error

    # 2. Проверка, что целевая переменная - бинарная
    binary_check, binary_error = check_binary_target(df, target_variable)
    if not binary_check:
        return False, binary_error

    # 3. Проверка целевой переменной на балансированность
    balance_check, balance_error = check_target_balance(df, target_variable)
    if not balance_check:
        return False, balance_error

    # если все требования соблюдены
    return True, None


def check_no_missing_values(df: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Проверка на пропущенные значения."""
    missing_counts = df.isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0]

    if len(columns_with_missing) > 0:
        column_details = ', '.join([
            f"{col}: {count} пропущено" 
            for col, count in columns_with_missing.items()
        ])
        return False, f"Набор данных содержит пропущенные значения: {column_details}"

    return True, None


def check_binary_target(
    df: pd.DataFrame, target_variable: str
) -> Tuple[bool, Optional[str]]:
    """Проверка, что целевая переменная - бинарная"""
    unique_values = df[target_variable].unique()

    if set(unique_values) == {0, 1}:
        return True, None

    # Попытка преобразовать в валидный бинарный признак (если строковая)
    try:
        numeric_values = pd.to_numeric(df[target_variable])
        unique_numeric = set(numeric_values.unique())

        if unique_numeric == {0, 1}:
            return True, None
        elif len(unique_numeric) == 2:
            error_msg = (
                f"Целевая переменная '{target_variable}' имеет значения "
                f"{unique_numeric} вместо требуемых 0/1. "
                f"Рассмотрите кодирование перед загрузкой."
            )
            return False, error_msg
        else:
            error_msg = (
                f"Целевая переменная '{target_variable}' имеет "
                f"{len(unique_numeric)} уникальных значений вместо 2."
            )
            return False, error_msg
    except Exception:
        error_msg = (
            f"Целевая переменная '{target_variable}' должна быть бинарной. "
            f"Найдены значения: {unique_values[:5]}..."
        )
        return False, error_msg


def check_target_balance(
    df: pd.DataFrame, target_variable: str, threshold: float = 0.2
) -> Tuple[bool, Optional[str]]:
    """Проверка целевой переменной на балансированность."""
    try:
        target_values = pd.to_numeric(df[target_variable])
        value_counts = target_values.value_counts(normalize=True)
        min_proportion = value_counts.min()

        if min_proportion < threshold:
            minority_class = value_counts.idxmin()
            error_msg = (
                f"Целевая переменная '{target_variable}' несбалансирована. "
                f"Класс {minority_class} представляет только "
                f"{min_proportion:.1%} данных. "
                f"Минимальное требуемое значение: {threshold:.1%}."
            )
            return False, error_msg

        return True, None
    except Exception:
        return False, f"Ошибка проверки баланса целевой переменной для '{target_variable}'"


def run_algorithm(type_algorithm_name, function_name, pipeline):
    print("RUN: run_algorithm")
    module_name = f'featurama.algorithms.{type_algorithm_name}.{function_name}'
    print('module_name:', module_name)
    module = importlib.import_module(module_name)
    cls = getattr(module, function_name)
    instance = cls(pipeline=pipeline)
    instance.run_method()


def build_pdf_report(output_stream, context):
    doc = SimpleDocTemplate(output_stream, pagesize=letter)
    styles = getSampleStyleSheet()

    # Настройка стилей
    pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
    styles['Title'].fontName = 'Arial'
    styles['Heading1'].fontName = 'Arial'
    styles['Heading2'].fontName = 'Arial'
    styles['Heading3'].fontName = 'Arial'
    styles['Normal'].fontName = 'Arial'

    styles['Title'].fontSize = 16
    styles['Heading1'].fontSize = 14
    styles['Heading2'].fontSize = 12
    styles['Heading3'].fontSize = 11
    styles['Normal'].fontSize = 10

    story = []

    # Заголовок
    story.append(Paragraph(f"Отчет по пайплайну №{context['pipeline'].id}", styles['Title']))
    story.append(Spacer(1, 12))

    # Информация
    story.append(Paragraph("Информация о пайплайне", styles['Heading1']))
    info = [
        f"Наименование набора данных: {context['dataset_name']}",
        f"Целевая переменная: {context['target_variable']}",
        f"Фильтрующий метод: {context['fs_filter']}",
        f"Метод-обертка: {context['fs_wrapper']}",
        f"Модель: {context['ml_model']}"
    ]
    for item in info:
        story.append(Paragraph(item, styles['Normal']))
    story.append(Spacer(1, 12))

    # Параметры
    story.append(Paragraph("Параметры методов", styles['Heading1']))

    if context['fs_filter_params_desc']:
        story.append(Paragraph(f"Параметры фильтрующего метода ({context['fs_filter']})", styles['Heading2']))
        for param in context['fs_filter_params_desc']:
            story.append(Paragraph(f"• <b>{param[0]}</b>: {param[1]}", styles['Normal']))
            story.append(Paragraph(f"  <i>{param[2]}</i>", styles['Normal']))
        story.append(Spacer(1, 6))

    if context['fs_wrapper_params_desc']:
        story.append(Paragraph(f"Параметры метода-обертки ({context['fs_wrapper']})", styles['Heading2']))
        for param in context['fs_wrapper_params_desc']:
            story.append(Paragraph(f"• <b>{param[0]}</b>: {param[1]}", styles['Normal']))
            story.append(Paragraph(f"  <i>{param[2]}</i>", styles['Normal']))
        story.append(Spacer(1, 6))

    if context['ml_model_params_desc']:
        story.append(Paragraph(f"Параметры модели ({context['ml_model']})", styles['Heading2']))
        for param in context['ml_model_params_desc']:
            story.append(Paragraph(f"• <b>{param[0]}</b>: {param[1]}", styles['Normal']))
            story.append(Paragraph(f"  <i>{param[2]}</i>", styles['Normal']))
        story.append(Spacer(1, 12))

    # Метрики
    story.append(Paragraph("Метрики модели", styles['Heading1']))
    metrics = [
        f"ROC-AUC: {context['ml_model_metrics']['roc_auc']}",
        f"Accuracy: {context['ml_model_metrics']['accuracy']}",
        f"F1-score: {context['ml_model_metrics']['f1_score']}",
        f"Precision: {context['ml_model_metrics']['precision']}",
        f"Recall: {context['ml_model_metrics']['recall']}"
    ]
    for item in metrics:
        story.append(Paragraph(item, styles['Normal']))
    story.append(Spacer(1, 12))

    # Признаки
    story.append(Paragraph("Выбранные признаки", styles['Heading1']))
    story.append(Paragraph("Начальные признаки:", styles['Heading2']))
    for feature in context['preliminarily_selected_features']:
        story.append(Paragraph(f"- {feature}", styles['Normal']))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Выбранные признаки:", styles['Heading2']))
    for feature in context['final_selected_features']:
        story.append(Paragraph(f"- {feature}", styles['Normal']))
    story.append(Spacer(1, 12))

    # SHAP графики
    story.append(Paragraph("SHAP анализ", styles['Heading1']))

    try:
        if context['shap_plot_global']:
            story.append(Paragraph("Глобальная важность признаков", styles['Heading2']))
            try:
                img = Image(
                    context['shap_plot_global'],
                    width=400,
                    height=300
                )
                story.append(img)
            except Exception as e:
                story.append(Paragraph(
                    f"Ошибка загрузки глобального SHAP графика: {str(e)}",
                    styles['Normal']
                ))
        else:
            story.append(Paragraph(
                "Нет доступного глобального SHAP графика",
                styles['Normal']
            ))
        story.append(Spacer(1, 12))

        if context['shap_plot_distribution']:
            story.append(Paragraph("Распределение важности признаков", styles['Heading2']))
            try:
                img = Image(
                    context['shap_plot_distribution'],
                    width=400,
                    height=300
                )
                story.append(img)
            except Exception as e:
                story.append(Paragraph(
                    f"Ошибка загрузки графика распределения SHAP: {str(e)}",
                    styles['Normal']
                ))
        else:
            story.append(Paragraph(
                "Нет доступного графика распределения SHAP",
                styles['Normal']
            ))
    except Exception as e:
        story.append(Paragraph(
            f"Ошибка доступа к SHAP анализу: {str(e)}",
            styles['Normal']
        ))

    doc.build(story)
