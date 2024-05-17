from typing import List

from django.http import HttpResponse
from django.shortcuts import render
from . import model
from matplotlib import pyplot as plt
import numpy as np
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def home(request):
    try:
        path1 = 'static/bar_chart.png'
        path2 = 'static/pie_chart.png'
        os.remove(path1)
        os.remove(path2)
    except:
        pass

    return render(request, 'index.html')


def manual(request):
    return render(request, 'manual.html')


def twitter_data_process(request):
    pre_result: List[bool]
    data, pre_result = model.Twitter_Data_Prediction()
    try:
        plot_chart(data, pre_result)
    except:
        pass
    return render(request, 'result.html', {
        'ex': data[0],
        'ex1': data[1],
        'ex2': data[2],
        'pr': pre_result[0],
        'pr1': pre_result[1],
        'pr2': pre_result[2]
    })


def manual_data_process(request):
    prediction = []
    user_input_text = request.POST.get('manual_type')
    prediction.append(model.Manual_Data_prediction(user_input_text))
    print("User Input is :", user_input_text)

    plot_chart(user_input_text, prediction)

    return render(request, 'result.html', {'ex': user_input_text, 'pr': prediction})


def plot_chart(three_data, prediction_result):
    plot_bar_chart(three_data, prediction_result)
    plot_pie_chart(three_data, prediction_result)


def plot_bar_chart(three_data, prediction_result):
    depressed = 0
    not_depressed = 0

    for i in prediction_result:
        if i:
            depressed += 1
        else:
            not_depressed += 1

    # fig = Figure()
    # canvas = FigureCanvas(fig)
    lab = ['Depressed', 'Not-Depressed']
    data = [depressed, not_depressed]
    plt.bar(lab, data, color=['red', 'green'])
    plt.title("Depresion Analysis")
    plt.xlabel("Type")
    plt.savefig('static/bar_chart.png')
    plt.close()


def plot_pie_chart(three_data, prediction_result):
    depressed = 0
    not_depressed = 0

    for i in prediction_result:
        if i:
            depressed += 1
        else:
            not_depressed += 1

    lab = ['Depressed', 'Not-Depressed']
    data = [depressed, not_depressed]
    # plt.pie(data, labels=lab, shadow=True)
    explode = (0.1, 0)
    plt.pie(data, explode=explode, labels=lab, autopct='%1.1f%%',shadow=True, startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.savefig('static/pie_chart.png')
    plt.close()
