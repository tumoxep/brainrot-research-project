import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scenedetect import open_video, SceneManager, ContentDetector
import cv2
import scipy.stats


def calculate_color_statistics(filePath):
    # Open the video
    cap = cv2.VideoCapture(filePath)
    # Initialize a list to store the saturation values
    saturation_values = []
    times = []

    # Loop over the frames
    while True:
        # Read a frame
        ret, frame = cap.read()

        # If the frame was not read successfully, break the loop
        if not ret:
            break

        # Convert the frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Calculate the average saturation
        avg_saturation = np.mean(hsv[:, :, 1])

        # Add the saturation value to the list
        saturation_values.append(avg_saturation)

        # Calculate the time for the current frame
        time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # Add the time to the list
        times.append(time)

    # Release the video
    cap.release()

    # Calculate the average saturation over all frames
    average_saturation = np.mean(saturation_values)

    # Calculate the median
    median = np.median(saturation_values)

    return (
        average_saturation,
        median,
        saturation_values,
        times,
    )


def plot_scene_changes(filePath):
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video = open_video(filePath)
    scene_manager.detect_scenes(video)

    # Get the list of scene changes.
    scene_list = scene_manager.get_scene_list()

    # Convert the scene change data into a pandas DataFrame.
    data = []
    for i, scene in enumerate(scene_list):
        start = scene[0].get_seconds()
        finish = scene[1].get_seconds()
        data.append(
            dict(Task="scene", Start=start, Finish=finish, Duration=finish - start)
        )

    df = pd.DataFrame(data)

    # Create a Gantt chart of the scene changes.
    timeline = px.bar(df, x="Duration", y="Task", color="Duration", height=200)

    # Calculate the total number of scenes and the total video duration.
    total_scenes = len(scene_list)
    total_duration = video.duration.get_seconds()

    # Calculate the average and peak scenes-per-minute.
    scene_times = [scene[0].get_seconds() for scene in scene_list]
    scene_intervals = np.diff(scene_times)
    scenes_per_minute = 60 / scene_intervals
    average_scenes_per_minute = np.mean(scenes_per_minute)
    peak_scenes_per_minute = np.max(scenes_per_minute)

    # Create indicator figures for the statistics
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

    (
        average_saturation,
        median,
        saturation_values,
        times,
    ) = calculate_color_statistics(filePath)

    # Create a line chart of the saturation values
    saturation_chart = px.line(
        x=times, y=saturation_values, title="Saturation Over Time"
    )

    # Create a DataFrame for the totals
    totals_data = {
        "Statistic": ["Total Scenes", "Total Duration", "Average Saturation", "Median Saturation"],
        "Value": [total_scenes, total_duration, average_saturation, median]
    }
    totals_df = pd.DataFrame(totals_data)

    # Return the figure and the statistics as a dictionary
    return (
        indicators,
        timeline,
        saturation_chart,
        totals_df,
    )


# Create the Gradio interface
inputs = gr.Video(show_label=False, sources=["upload"])
outputs = [
    gr.Plot(show_label=False),
    gr.Plot(show_label=False),
    gr.Plot(show_label=False),
    gr.Dataframe(show_label=False),
]
interface = gr.Interface(
    fn=plot_scene_changes,
    inputs=inputs,
    outputs=outputs,
    allow_flagging="never",
    css=".upload-container .wrap {color: rgba(0,0,0,0)} .upload-container .wrap >*:not(.icon-wrap) {color: rgba(0,0,0,0)} .upload-container .wrap .icon-wrap {color: white}",
)
interface.launch()
