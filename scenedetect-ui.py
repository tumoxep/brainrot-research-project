import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scenedetect import open_video, SceneManager, ContentDetector


def plot_scene_changes(filePath):
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video = open_video(filePath)
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()

    data = []
    for i, scene in enumerate(scene_list):
        start = scene[0].get_seconds()
        finish = scene[1].get_seconds()
        data.append(
            dict(Task="scene", Start=start, Finish=finish, Duration=finish - start)
        )

    df = pd.DataFrame(data)

    timeline = px.bar(df, x="Duration", y="Task", color="Duration", height=200)

    total_scenes = len(scene_list)
    total_duration = video.duration.get_seconds()

    scene_times = [scene[0].get_seconds() for scene in scene_list]
    scene_intervals = np.diff(scene_times)
    scenes_per_minute = 60 / scene_intervals
    average_scenes_per_minute = np.mean(scenes_per_minute)
    peak_scenes_per_minute = np.max(scenes_per_minute)

    indicators = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [
                {"type": "indicator", "l": 0.1, "r": 0.1, "t": 0.2, "b": 0.1},
                {"type": "indicator", "l": 0.1, "r": 0.1, "t": 0.2, "b": 0.1},
            ],
        ],
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    indicators.update_layout(height=200)
    indicators.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=average_scenes_per_minute,
            title={"text": "Average Scenes-per-Minute"},
            gauge={
                "axis": {"range": [None, 60]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 15], "color": "#90ee90"},
                    {"range": [15, 45], "color": "#FF7F7F"},
                    {"range": [45, 60], "color": "red"},
                ],
            },
        ),
        row=1,
        col=1,
    )
    indicators.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=peak_scenes_per_minute,
            title={"text": "Peak Scenes-per-Minute"},
            gauge={
                "axis": {"range": [None, 150]},
                "bar": {"color": "black"},
                "steps": [
                    {"range": [0, 75], "color": "#90ee90"},
                    {"range": [75, 120], "color": "#FF7F7F"},
                    {"range": [120, 150], "color": "red"},
                ],
            },
        ),
        row=1,
        col=2,
    )

    totals = f"<h1>Total Scenes: {total_scenes}</h1><h1>Total Duration: {total_duration:.2f} seconds</h1>"

    return indicators, timeline, totals


inputs = gr.Video(show_label=False, sources=["upload"])
outputs = [gr.Plot(show_label=False), gr.Plot(show_label=False), gr.HTML()]
interface = gr.Interface(
    fn=plot_scene_changes, inputs=inputs, outputs=outputs, allow_flagging="never"
)
interface.launch()
