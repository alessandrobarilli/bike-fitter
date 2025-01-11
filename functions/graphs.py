import plotly.graph_objects as go
import numpy as np


def gauge_chart(
        var: int, 
        valid_range: list,
        optimal_range: list,
        dimensions: tuple = (600, 20),
        title: str = ""
) -> go.Figure:
    
    color_range = "lightgreen" if optimal_range[0] <= np.mean(var) <= optimal_range[1] else "lightcoral"
    color_bullet = "darkgreen" if optimal_range[0] <= np.mean(var) <= optimal_range[1] else "darkred"

    # Create the gauge chart
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=int(np.mean(var)),
        title={'text': title},
        gauge={
            'shape': "bullet",
            'bar': {'color': color_bullet},
            'axis': {'range': valid_range},
            'steps': [
                {'range': optimal_range, 'color': color_range},
            ],
        })
    )

    # Adjust the width and height of the plot
    gauge_fig.update_layout(
        autosize=False,
        margin=dict(t=0, b=0),
        width=dimensions[0],
        height=dimensions[1],
        font = {'color': color_range},
    )

    return gauge_fig
