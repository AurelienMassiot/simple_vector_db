import pandas as pd
import seaborn as sns

colors = ["#EE0023", "#0E2356", "#00AD00", "#00A3BE", "#00A795", "#FFE300", "#FF6200", "#9A78F0", "#9FA7BB", "#F899A7"]
CUSTOM_PALETTE = sns.color_palette(colors)


def plot_results(data: pd.DataFrame, x: str, y: str, path_to_save: str, hue: str = None):
    sns_plot = sns.scatterplot(data=data, x=x, y=y, hue=hue, palette=CUSTOM_PALETTE)
    fig = sns_plot.get_figure()
    fig.savefig(path_to_save)
