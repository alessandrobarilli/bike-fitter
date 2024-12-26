import plotly.graph_objects as go
import numpy as np


def gauge_chart(
        var: int, 
        valid_range: list,
        optimal_range: list,
        dimensions: tuple = (600, 20),
        title: str = ""
) -> go.Figure:
    # Create the gauge chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(np.mean(var)),
        title={'text': title},
        gauge={
            'shape': "bullet",
            'axis': {'range': valid_range},
            'steps': [
                {'range': optimal_range, 'color': "lightgreen"},
            ],
        })
    )

    # Adjust the width and height of the plot
    gauge_fig.update_layout(
        autosize=False,
        margin=dict(t=0, b=0),
        width=dimensions[0],
        height=dimensions[1],
    )

    return gauge_fig
