# Plotly: sibur_preparing/completed/plotly_interactive.py, ViTya/visualization.ipynb
# pip install plotly pandas

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "notebook"  # notebook / vscode / png
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
temp = df[df["day"] == "Sun"]

# 1. Express: x/y — колонки, color — группы, size — размер точек
# barmode: overlay / group / stack; facet_col/row — сетка; hover_data — подсказки
px.histogram(temp, x="total_bill", color="sex", title="Histogram").show()
px.box(temp, x="sex", y="total_bill", title="Boxplot").show()
px.scatter(temp, x="total_bill", y="tip", color="sex", size="size", title="Scatter").show()
px.bar(temp, x="sex", y="tip", color="sex", title="Bar").show()
px.histogram(temp, x=["total_bill", "tip"], barmode="overlay", title="Несколько метрик").show()

# 2. Graph Objects: Figure → add_trace → update_layout → show
fig = go.Figure()
fig.add_trace(go.Histogram(x=temp["total_bill"], name="total_bill"))
fig.add_trace(go.Histogram(x=temp["tip"], name="tip", opacity=0.5))
fig.update_layout(barmode="overlay", title="Два распределения")
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=temp["total_bill"], y=temp["tip"], mode="markers", name="tips"))
fig.update_layout(title="Scatter", xaxis_title="total_bill", yaxis_title="tip")
fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=temp.index, y=temp["total_bill"], mode="lines", name="total_bill"))
fig.update_layout(title="Line")
fig.show()

fig = go.Figure()
fig.add_trace(go.Bar(x=temp["sex"], y=temp["tip"], name="tip"))
fig.update_layout(title="Bar")
fig.show()

fig = go.Figure()
fig.add_trace(go.Box(y=temp["total_bill"], name="total_bill"))
fig.add_trace(go.Box(y=temp["tip"], name="tip"))
fig.update_layout(title="Boxplot")
fig.show()

# 3. Сетка: row/col указывают ячейку для каждого trace
fig = make_subplots(rows=2, cols=2, subplot_titles=["Scatter", "Line", "Bar", "Boxplot"])
fig.add_trace(go.Scatter(x=temp["total_bill"], y=temp["tip"], mode="markers", name="tips"), row=1, col=1)
fig.add_trace(go.Scatter(x=temp.index, y=temp["total_bill"], mode="lines+markers", name="total_bill"), row=1, col=2)
fig.add_trace(go.Bar(x=temp["sex"], y=temp["tip"], name="tip"), row=2, col=1)
fig.add_trace(go.Box(y=temp["total_bill"], name="total_bill"), row=2, col=2)
fig.update_layout(title="Several Graphics", height=700)
fig.show()

# 4. Переключатели из ViTya — на тех же колонках tips
# restyle: меняем данные одного trace
names = ["total_bill", "tip"]
fig = go.Figure(go.Bar(x=temp.index, y=temp[names[0]]))
buttonlist = [
    dict(args=["y", [temp[label]]], label=label, method="restyle")
    for label in names
]
fig.update_layout(updatemenus=[go.layout.Updatemenu(buttons=buttonlist, direction="down")])
fig.show()

# update: показываем один из заранее добавленных traces
fig = go.Figure()
for i, name in enumerate(names):
    fig.add_trace(go.Scatter(x=temp.index, y=temp[name], mode="lines", name=name, visible=i == 0))
fig.update_layout(updatemenus=[{
    "buttons": [
        {"label": name, "method": "update", "args": [{"visible": [i == j for j in range(len(names))]}]}
        for i, name in enumerate(names)
    ],
    "direction": "down",
}])
fig.show()
