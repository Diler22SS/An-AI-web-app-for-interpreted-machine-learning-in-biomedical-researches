# VIEWS.PY
import os
import json
from io import BytesIO
import base64
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpRequest, HttpResponse
from .models import (
    Pipeline, FS_Filter, FS_Wrapper, ML_Model
)
from .forms import (
    FileUploadForm, TargetVariableForm, FeatureSelectionForm
)
from .utils import (
    read_dataset_file, validate_dataset, run_algorithm, build_pdf_report
)


def pipelines(request: HttpRequest) -> HttpResponse:
    """ Список всех пайплайнов и создание нового пайплайна """
    if request.method == 'POST':
        new_pipeline = Pipeline.objects.create()
        return redirect('featurama:upload_file', pipeline_id=new_pipeline.pk)

    pipelines = Pipeline.objects.all()

    return render(
        request,
        'featurama/pipelines.html',
        {'pipelines': pipelines}
    )


def upload_file(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """ Обработка загруженного файла """
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    error = None

    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                uploaded_file = form.cleaned_data['dataset_file']
                filename = os.path.splitext(uploaded_file.name)[0]
                df = read_dataset_file(uploaded_file)

                pipeline.filename = filename
                pipeline.save_dataframe(df)
                pipeline.save()

                return redirect('featurama:select_target_variable', pipeline_id=pipeline_id)
            except pd.errors.ParserError:
                error = "Ошибка чтения файла. Проверьте формат данных."
            except Exception as e:
                error = f"Неизвестная ошибка при обработке файла. ({e})"
        else:
            error = form.errors.get('dataset_file', ['Ошибка загрузки файла'])[0]
    else:
        form = FileUploadForm()

    return render(
        request,
        'featurama/upload_file.html',
        {
            'pipeline': pipeline,
            'error': error
        }
    )


def select_target_variable(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """ Обработка выбора целевой переменной """
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    error = None

    df = pipeline.get_dataframe()
    if df.empty:
        print("Ошибка при загрузке набора данных")
        return redirect('featurama:upload_file', pipeline_id=pipeline_id)

    filename = pipeline.filename
    features = df.columns.tolist()

    if request.method == 'POST':
        form = TargetVariableForm(request.POST, features=features)
        if form.is_valid():
            try:
                target_variable = form.cleaned_data['target_variable']
                pipeline.target_variable = target_variable
                pipeline.save()

                return redirect('featurama:pre_select_features', pipeline_id=pipeline_id)
            except Exception as e:
                error = f"Ошибка сохранения. ({e})"
        else:
            error = "Пожалуйста, выберите корректную целевую переменную."
    else:
        form = TargetVariableForm(features=features)

    return render(
        request,
        'featurama/select_target_variable.html',
        {
            'pipeline': pipeline,
            'dataset_name': filename,
            'form': form,
            'error': error
        }
    )


def pre_select_features(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """ Обработка предварительного выбора набора признаков """
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    error = None

    df = pipeline.get_dataframe()
    target_variable = pipeline.target_variable
    if target_variable is None:
        print("Целевая переменная не выбрана")
        return redirect('featurama:select_target_variable', pipeline_id=pipeline_id)

    filename = pipeline.filename
    features = df.columns.tolist()

    if request.method == 'POST':
        form = FeatureSelectionForm(
            request.POST,
            features=features,
            target_variable=target_variable
            )
        if form.is_valid():
            selected_features = form.cleaned_data['selected_features']
            is_valid, message = validate_dataset(
                df, target_variable, selected_features
            )
            if not is_valid:
                error = message
            else:
                try:
                    pipeline.preliminarily_selected_features = selected_features
                    pipeline.save()

                    return redirect(
                        'featurama:configure_pipeline',
                        pipeline_id=pipeline.pk
                    )
                except Exception as e:
                    error = f"Не удалось сохранить выбранные признаки. ({e})"
        else:
            error = form.errors.get('selected_features', ['Некорректный выбор'])[0]
    else:
        form = FeatureSelectionForm(
            features=features,
            target_variable=target_variable
            )
    return render(
        request,
        'featurama/pre_select_features.html',
        {
            'pipeline': pipeline,
            'dataset_name': filename,
            'target_variable': target_variable,
            'feature_form': form,
            'error': error
        }
    )


def configure_pipeline(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """ Обработка отбора признаков """
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    error = None

    filename = pipeline.filename
    target_variable = pipeline.target_variable

    preliminarily_selected_features = pipeline.preliminarily_selected_features
    if preliminarily_selected_features is None:
        print("Предварительно набор признаков не выбран")
        return redirect('featurama:pre_select_features', pipeline_id=pipeline_id)

    if request.method == 'POST':
        filter_method = request.POST.get('filter_method')
        wrapper_method = request.POST.get('wrapper_method')

        filter_params = {
            key.replace('filter_', ''): value
            for key, value in request.POST.items()
            if key.startswith('filter_') and key != 'filter_method'
        }

        wrapper_params = {
            key.replace('wrapper_', ''): value
            for key, value in request.POST.items()
            if key.startswith('wrapper_') and key != 'wrapper_method'
        }

        if not filter_method or not filter_params:
            error = "Пожалуйста, настройте фильтрационный метод."
        elif not wrapper_method or not wrapper_params:
            error = "Пожалуйста, настройте метод-обертку."
        else:
            try:
                print("Filter Method:", filter_method)
                print("Filter Params:", filter_params)
                print("Wrapper Method:", wrapper_method)
                print("Wrapper Params:", wrapper_params)

                fs_filter = FS_Filter.objects.get(function_name=filter_method)
                fs_wrapper = FS_Wrapper.objects.get(function_name=wrapper_method)

                pipeline.fs_filter = fs_filter
                pipeline.fs_filter_parameters = filter_params
                pipeline.fs_wrapper = fs_wrapper
                pipeline.fs_wrapper_parameters = wrapper_params
                pipeline.save()

                run_algorithm('fs_filter', filter_method, pipeline)
                run_algorithm('fs_wrapper', wrapper_method, pipeline)

                return redirect(
                    'featurama:manual_feature_selection',
                    pipeline_id=pipeline.pk
                )
            except FS_Filter.DoesNotExist:
                error = "Выбранный фильтрационный метод не найден."
            except FS_Wrapper.DoesNotExist:
                error = "Выбранный метод-обёртка не найден."
            except Exception as e:
                error = f"Произошла ошибка при сохранении настроек. ({e})"

    filters = FS_Filter.objects.all()
    wrappers = FS_Wrapper.objects.all()

    return render(
        request,
        'featurama/configure_pipeline.html',
        {
            'pipeline': pipeline,
            'dataset_name': filename,
            'target_variable': target_variable,
            'preliminarily_selected_features': preliminarily_selected_features,
            'filters': filters,
            'wrappers': wrappers,
            'error': error
        }
    )


def manual_feature_selection(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """ Ручная корректировка отобранного набора признаков """
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    error = None

    filename = pipeline.filename
    target_variable = pipeline.target_variable
    preliminarily_selected_features = pipeline.preliminarily_selected_features
    fs_filter_selected_features = pipeline.fs_filter_selected_features
    fs_wrapper_selected_features = pipeline.fs_wrapper_selected_features

    final_selected_features = fs_wrapper_selected_features

    if fs_wrapper_selected_features is None:
        print(f"Отобранный набор признаков пуст")
        return redirect('featurama:configure_pipeline', pipeline_id=pipeline_id)

    if request.method == 'POST':
        final_selected_features = request.POST.getlist('selected_features')
        model_method = request.POST.get('model_method')

        model_params = {
            key.replace('model_', ''): value
            for key, value in request.POST.items()
            if key.startswith('model_') and key != 'model_method'
        }

        if not final_selected_features:
            error = 'Пожалуйста, выберите хотя бы один признак.'
        elif not model_method or not model_params:
            error = "Пожалуйста, настройте модель."
        else:
            try:
                print("Manually selected features:", final_selected_features)
                print("Model Method:", model_method)
                print("Model Params:", model_params)

                ml_model = ML_Model.objects.get(function_name=model_method)

                pipeline.final_selected_features = final_selected_features
                pipeline.ml_model = ml_model
                pipeline.ml_model_parameters = model_params
                pipeline.save()

                run_algorithm('ml_models', model_method, pipeline)

                return redirect(
                    'featurama:results_summary',
                    pipeline_id=pipeline.pk
                )
            except ML_Model.DoesNotExist:
                error = "Выбранный метод машинного обучения не найден."
            except Exception as e:
                error = f"Произошла ошибка при сохранении настроек. ({e})"

    models = ML_Model.objects.all()

    return render(
        request,
        'featurama/manual_feature_selection.html',
        {
            'pipeline': pipeline,
            'dataset_name': filename,
            'target_variable': target_variable,
            'user_features': preliminarily_selected_features,
            'filtered_features': fs_filter_selected_features,
            'wrapped_features': fs_wrapper_selected_features,
            'selected_features': final_selected_features,
            'models': models,
            'error': error
        }
    )


def _get_param_description(parameters, method_instance):
    """
    Возвращает список параметров методов.
    """
    if not parameters or not method_instance:
        return []

    adjustable_params = getattr(method_instance, 'adjustable_parameters', {})
    result = []
    for key, value in dict(parameters).items():
        description = adjustable_params.get(key, {}).get('title', 'Нет описания')
        result.append([key, value, description])
    return result


def results_summary(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """ Отобразить результаты работы пайплайна """
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    error = None

    filename = pipeline.filename
    target_variable = pipeline.target_variable
    preliminarily_selected_features = pipeline.preliminarily_selected_features

    fs_filter = pipeline.fs_filter
    fs_filter_selected_features = pipeline.fs_filter_selected_features
    fs_filter_params_desc = _get_param_description(pipeline.fs_filter_parameters, pipeline.fs_filter)

    fs_wrapper = pipeline.fs_wrapper
    fs_wrapper_selected_features = pipeline.fs_wrapper_selected_features
    fs_wrapper_params_desc = _get_param_description(pipeline.fs_wrapper_parameters, pipeline.fs_wrapper)

    final_selected_features = pipeline.final_selected_features

    ml_model = pipeline.ml_model
    ml_model_params_desc = _get_param_description(pipeline.ml_model_parameters, pipeline.ml_model)
    ml_model_metrics = json.loads(pipeline.ml_model_metrics)
    ml_model_shap_values = json.loads(pipeline.ml_model_shap_values)
    shap_plot_global = ml_model_shap_values['global_shap_plot']
    shap_plot_distribution = ml_model_shap_values['distribution_shap_plot']

    print(filename)
    print(target_variable)
    print(preliminarily_selected_features)
    print(fs_filter, fs_filter.adjustable_parameters)
    print(fs_filter_selected_features)
    print(fs_wrapper, fs_wrapper.adjustable_parameters)
    print(fs_wrapper_selected_features)
    print(final_selected_features)
    print(ml_model, ml_model.adjustable_parameters)
    print(ml_model_metrics)
    # print(ml_model_shap_values)

    if ml_model_metrics is None or ml_model_shap_values is None:
        print("Модель не обучена")
        return redirect('featurama:configure_pipeline', pipeline_id=pipeline_id)

    context = {
        'pipeline': pipeline,
        'dataset_name': filename,
        'target_variable': target_variable,
        'preliminarily_selected_features': preliminarily_selected_features,
        'fs_filter': fs_filter,
        'fs_filter_params_desc': fs_filter_params_desc,
        'fs_filter_selected_features': fs_filter_selected_features,
        'fs_wrapper': fs_wrapper,
        'fs_wrapper_params_desc': fs_wrapper_params_desc,
        'fs_wrapper_selected_features': fs_wrapper_selected_features,
        'final_selected_features': final_selected_features,
        'ml_model': ml_model,
        'ml_model_params_desc': ml_model_params_desc,
        'ml_model_metrics': ml_model_metrics,
        'shap_plot_global': shap_plot_global,
        'shap_plot_distribution': shap_plot_distribution,
        'error': error
    }

    return render(request, 'featurama/results_summary.html', context)


def delete_pipeline(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """ Удаление пайплайна """
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)
    pipeline.delete()

    return redirect('featurama:pipelines')


def export_report(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    """ Формирование отчета о результатах работы пайплайна """
    pipeline = get_object_or_404(Pipeline, pk=pipeline_id)

    filename = pipeline.filename
    target_variable = pipeline.target_variable
    preliminarily_selected_features = pipeline.preliminarily_selected_features

    fs_filter = pipeline.fs_filter
    fs_filter_params_desc = _get_param_description(pipeline.fs_filter_parameters, pipeline.fs_filter)

    fs_wrapper = pipeline.fs_wrapper
    fs_wrapper_params_desc = _get_param_description(pipeline.fs_wrapper_parameters, pipeline.fs_wrapper)

    final_selected_features = pipeline.final_selected_features

    ml_model = pipeline.ml_model
    ml_model_params_desc = _get_param_description(pipeline.ml_model_parameters, pipeline.ml_model)
    ml_model_metrics = json.loads(pipeline.ml_model_metrics)
    ml_model_shap_values = json.loads(pipeline.ml_model_shap_values)
    shap_plot_global = BytesIO(base64.b64decode(ml_model_shap_values['global_shap_plot']))
    shap_plot_distribution = BytesIO(base64.b64decode(ml_model_shap_values['distribution_shap_plot']))

    context = {
        'pipeline': pipeline,
        'dataset_name': filename,
        'target_variable': target_variable,
        'preliminarily_selected_features': preliminarily_selected_features,
        'final_selected_features': final_selected_features,

        'fs_filter': fs_filter,
        'fs_filter_params_desc': fs_filter_params_desc,

        'fs_wrapper': fs_wrapper,
        'fs_wrapper_params_desc': fs_wrapper_params_desc,

        'ml_model': ml_model,
        'ml_model_params_desc': ml_model_params_desc,
        'ml_model_metrics': ml_model_metrics,

        'shap_plot_global': shap_plot_global,
        'shap_plot_distribution': shap_plot_distribution,
    }

    # Формируем PDF
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="pipeline_{pipeline_id}_report.pdf"'

    try:
        build_pdf_report(response, context)
    except Exception as e:
        print(f"Ошибка при построении PDF. ({e})")
        return HttpResponse("Ошибка при формировании отчёта", status=500)

    return response
