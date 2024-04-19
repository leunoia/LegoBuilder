import panel as pn
import numpy as np
import pandas as pd

def create_plot():
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    df = pd.DataFrame({'x': x, 'y': y})
    return pn.pane.DataFrame(df, width=400, height=300)

pn.extension(design="material", sizing_mode="strength_width") 
# is used to load and configure Panel extensions and add HTML/CSS styling 
plot = create_plot()

plot.servable()
