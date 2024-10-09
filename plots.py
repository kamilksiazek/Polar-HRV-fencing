import os
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from utils import load_arbitrary_dataframe


def plot_RR_interval(dataframe,
                     x_column_name,
                     y_column_name,
                     anomalies=None,
                     saving_folder='./',
                     title=None,
                     name=None):
    """
    Plot one-dimensional, i.e. RR-intervals signal.

    Arguments:
    ----------
      *dataframe*: (Pandas Dataframe) contains data which should be plotted
      *x_column_name*:
      *y_column_name*:
      *anomalies* (list or Numpy array) optional anomalies for plot
      *saving_folder* (string) optional, custom folder for saving
      *name* (string) optional, custom filename for saving
    """
    fig, ax = plt.subplots(figsize=(25, 6))
    data = dataframe.copy()
    data[x_column_name] = pd.to_datetime(
        data[x_column_name]
    ).dt.time
    if name is None:
        timestamp = data.iloc[0]['Phone timestamp']
        name = f'RR_interval_plot_{timestamp}'
    if saving_folder != './':
        os.makedirs(saving_folder, exist_ok=True)
        name = f'{name}'
    data.plot(
        x=x_column_name,
        y=y_column_name,
        ax=ax,
        kind='line',
        color='red',
        linewidth=0.8,
        xlabel=x_column_name,
        ylabel=y_column_name,
    )
    if anomalies is not None:
        times_of_anomalies = data.loc[anomalies]["Phone timestamp"]
        times_of_anomalies.to_csv(
            f'{name}.csv',
            index=False
        )
        ymin, ymax = ax.get_ylim()
        if ymax > 1200.:
            ymax = 1200.
        if ymin < 0.:
            ymin = 0.
        plt.ylim([ymin, ymax])
        ax.vlines(
            x=times_of_anomalies.values,
            ymin=ymin,
            ymax=ymax-1,
            linewidth=0.8
        )
        data = data.drop(anomalies)

    delta = datetime.timedelta(minutes=5)
    xticks = [data.iloc[0]["Phone timestamp"]]
    while xticks[-1] <= data.iloc[-1]["Phone timestamp"]:
        dummy_date = datetime.datetime(2023, 12, 1)
        date_and_time = datetime.datetime.combine(dummy_date, xticks[-1])
        new_datetime = date_and_time + delta
        xticks.append(new_datetime.time())
    ax.set_xticks(xticks)

    plt.xticks(rotation=90, fontsize=8)
    if title is not None:
        plt.title(title)
    plt.savefig(f'{name}.pdf', bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()


def plot_HRV_and_battles_results(HRV_windows_values,
                                 median_timestamps,
                                 main_folder,
                                 person,
                                 saving_folder,
                                 method,
                                 limited=False,
                                 show_entering=False,
                                 lower_bound="",
                                 upper_bound=""):
    """
    Plot mean HRV values with marked lost/gained points during battles.

    Arguments:
    ----------
        *HRV_windows_values*: (Numpy array) contains mean HRV values during
                                consecutive time windows
        *median_timestamps*: (Numpy array) contains median timestamps for
                            corresponding time windows from
                            *HRV_windows_values*
        *main_folder*: (string) folder containing the description of battles
        *person*: (string) identification number of the considered person
        *method*: (string) method of HRV calculation for the label of y-axis
        *saving_folder*: (string) folder for saving plots
        *limited*: (Boolean) optional argument defining whether a subset of
                    HRV values will be presented (default: False)
        *show_entering*: (Boolean) optional argument defining whether entering
                         the mat and beginning of the battle will be displayed
                         (default: False)
        *lower_bound*: (string) defines a lower bound of the subset of HRV
                        values for presentation (default: empty string)
        *upper_bound*: (string) defines an upper bound of the subset of HRV
                        values for presentation (default: empty string)
    """
    # Plot HRV vs battles results
    battles = load_arbitrary_dataframe(
        f'{main_folder}{person}/',
        name=f'battles_{person}.csv'
    )
    if show_entering:
        entering = load_arbitrary_dataframe(
            f'{main_folder}{person}/',
            name=f'starts_{person}.csv'
        )
    x_column = 'timestamp'
    y_column = 'HRV'
    HRV_dataframe = pd.DataFrame(
        {x_column: median_timestamps,
         y_column: HRV_windows_values}
    )
    HRV_dataframe[x_column] = pd.to_datetime(
        HRV_dataframe[x_column]
    ).dt.time
    if limited:
        fig, ax = plt.subplots()
    else:
        fig, ax = plt.subplots(figsize=(15, 3))
    sns.set_theme(style="whitegrid")
    HRV_dataframe.plot(
        x=x_column,
        y=y_column,
        ax=ax,
        kind='line',
        color='blue',
        linewidth=1,
        legend=False,
        xlabel=x_column,
        ylabel=f'HRV ({method})'
    )
    if limited:
        # When x-axis is limited it is also better to limit the range of y-axis
        values = HRV_dataframe.loc[
            (HRV_dataframe['timestamp'] >= datetime.datetime.strptime(
                lower_bound, '%H:%M:%S').time()) &
            (HRV_dataframe['timestamp'] <= datetime.datetime.strptime(
                upper_bound, '%H:%M:%S').time())]['HRV'].values
        min_value, max_value = np.min(values), np.max(values)
        ymin = min_value - 0.1 * min_value
        ymax = max_value + 0.1 * max_value
    else:
        ymin, ymax = ax.get_ylim()
    plt.ylim([ymin, ymax])

    # Change ticks to ensure that we they are starting from "full minutes"
    delta = 1 if limited else 5
    delta = datetime.timedelta(minutes=delta)
    xticks = [HRV_dataframe.iloc[0]["timestamp"].replace(
        second=0, microsecond=0)]
    while xticks[-1] <= HRV_dataframe.iloc[-1]["timestamp"]:
        dummy_date = datetime.datetime(2023, 12, 1)
        date_and_time = datetime.datetime.combine(dummy_date, xticks[-1])
        new_datetime = date_and_time + delta
        xticks.append(new_datetime.time())
    ax.set_xticks(xticks)

    # Plot lost and gained points
    lost_gained_points = [[-1, 'red', 'lost point'],
                          [1, 'green', 'gained point']]
    for value, color, category in lost_gained_points:
        points = battles.loc[battles['value'] == value]
        ax.vlines(
            x=points['time'].values,
            ymin=ymin,
            ymax=ymax,
            linewidth=1,
            color=color,
            label=category
        )

    # Show additional information, i.e. entering the mat and starting
    # a battle.
    if show_entering:
        entering_battles = {
            'Entering': ['coral', 'entering'],
            'Start': ['dodgerblue', 'battle start']
        }
        additional_patches = []
        for key in list(entering_battles.keys()):
            ax.vlines(
                x=entering[key].values,
                ymin=ymin,
                ymax=ymax,
                linewidth=1,
                color=entering_battles[key][0],
                label=entering_battles[key][1],
            )
            additional_patches.append(
                mpatches.Patch(color=entering_battles[key][0],
                               label=entering_battles[key][1])
            )

    # Add plot range
    lost_patch = mpatches.Patch(color='red', label='lost point')
    gained_patch = mpatches.Patch(color='green', label='gained point')
    if show_entering:
        plt.legend(handles=[lost_patch, gained_patch, *additional_patches])
    else:
        plt.legend(handles=[lost_patch, gained_patch])
    if limited:
        plt.xlim(lower_bound, upper_bound)
    plt.xticks(rotation=90)
    plt.title(f'HRV values depending on time for {person}')

    name = f'HRV_{method}_battles_{person}_'
    if show_entering:
        name = f'{name}_entering_'
    if limited:
        name = (f'{name}{lower_bound.replace(":", "")}_'
                f'{upper_bound.replace(":", "")}.png')
    else:
        name = (f'{name}'
                f'{HRV_dataframe.iloc[0]["timestamp"].strftime("%H%M%S")}.pdf')

    if saving_folder != './':
        os.makedirs(saving_folder, exist_ok=True)
    name = f'{saving_folder}{name}'
    plt.savefig(f'{name}', bbox_inches='tight', dpi=400)
    plt.show()
    plt.close()


if __name__ == "__main__":
    pass
