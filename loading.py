
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import date
import datetime
from plots import (
    plot_HRV_and_battles_results,
    plot_RR_interval,
)
from utils import load_arbitrary_dataframe

# Load path to files with the previous project
###
cur_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_path, "HRV_patients_analysis"))
###

from HRV_patients_analysis.utils_loading import concat_files
from HRV_patients_analysis.utils_preprocessing import (
    convert_absolute_time_to_timestamps_from_given_timestamp,
    remove_adjacent_beats,
    remove_consecutive_beats_after_holes,
    remove_first_and_last_indices,
    remove_negative_timestamps,
    select_indices_to_filtering
)
from HRV_patients_analysis.HRV_calculation import (
    calculate_HRV_in_windows
)


def prepare_dataframe_for_single_person(dataframe):
    """
    Prepare basic time transformations in a given Pandas dataframe.
    """
    dataframe["Phone timestamp"] = pd.to_datetime(dataframe["Phone timestamp"])
    initial_timestamp = dataframe.iloc[0]["Phone timestamp"]
    dataframe = convert_absolute_time_to_timestamps_from_given_timestamp(
        dataframe, initial_timestamp
    )
    return dataframe


if __name__ == "__main__":
    main_folder = '../Data/'
    persons = [
        '9A4FFC2F',
        '9F865C29',
        '944F832B',
        '968A6E29',
        '86073D21',
        'A8438D20'
    ]
    battles_hours = [
        [['18:39:00', '18:48:30'], ['19:02:30', '19:13:00'],
         ['19:17:30', '19:26:00'], ['19:40:00', '19:48:00']],
        [['18:39:30', '18:51:00'], ['18:53:00', '19:01:30'],
         ['19:02:30', '19:12:30'], ['19:26:30', '19:42:00']],
        [['18:39:30', '18:51:00'], ['18:56:00', '19:03:00']],
        [['18:39:30', '18:52:00'], ['18:53:00', '19:01:30'],
         ['19:18:00', '19:25:30'], ['19:31:00', '19:38:30']],
        [['18:40:00', '18:51:30'], ['18:56:00', '19:03:00'],
         ['19:13:30', '19:21:30'],
         ['19:27:00', '19:31:00'], ['19:32:30', '19:42:00'],
         ['19:49:30', '19:57:30']],
        [['18:39:00', '18:48:00'], ['19:13:30', '19:21:30'],
         ['19:22:00', '19:28:00'], ['19:36:00', '19:39:30']]
    ]
    parameters = {
        'cut_time_from_start': '45 seconds',
        'cut_time_before_finish': '45 seconds',
        'threshold_for_hole_duration': '30 seconds',
        'time_after_hole_for_removing': '10 seconds',
        'adjacent_beats_for_removing': '1 seconds',
        'step_frequency': '15 seconds',
        'window_size': '5 min',
        'method': 'SDNN',  # options: 'RMSSD', 'SDNN'
    }
    saving_folder = f'./Plots/{parameters["method"]}/'
    os.makedirs(saving_folder, exist_ok=True)
    datatype = 'RR'
    for person in persons:
        path = f'{main_folder}{person}/'
        concat_files(path, datatype, save=True)

    filenames = {
        '9F865C29': 'Polar_H10_9F865C29_20230329_182142_RR_full.txt',
        '9A4FFC2F': 'Polar_H10_9A4FFC2F_20230329_181746_RR_full.txt',
        '944F832B': 'Polar_H10_944F832B_20230329_182144_RR_full.txt',
        '968A6E29': 'Polar_H10_968A6E29_20230329_181422_RR_full.txt',
        '86073D21': 'Polar_H10_86073D21_20230329_181349_RR_full.txt',
        'A8438D20': 'Polar_H10_A8438D20_20230329_181740_RR_full.txt'
    }
    for person, battles in zip(persons, battles_hours):
        name = filenames[person]
        dataframe = load_arbitrary_dataframe(
            f'{main_folder}{person}/',
            name=name
        )
        dataframe = prepare_dataframe_for_single_person(dataframe)

        x_column_name = 'Phone timestamp'
        y_column_name = 'RR-interval [ms]'
        plot_RR_interval(dataframe,
                         x_column_name,
                         y_column_name,
                         saving_folder=saving_folder,
                         title=f'RR-interval plot for {person}',
                         name=f'{person}_full_RR_intervals')

        # Remove negative timedeltas. In some cases particular
        # measurements are obtained with delay
        data = remove_negative_timestamps(dataframe)

        # Remove first and last few measurements as a typical source
        # of anomalies
        data = remove_first_and_last_indices(
            data,
            parameters['cut_time_from_start'],
            parameters['cut_time_before_finish']
        )

        # Remove some measurements after longer holes in the dataset
        data = remove_consecutive_beats_after_holes(
            data,
            parameters['threshold_for_hole_duration'],
            parameters['time_after_hole_for_removing']
        )

        data = data.reset_index(drop=True)
        # Prepare Discrete Wavelet Transform
        DWT_coefficients, filtered_indices = select_indices_to_filtering(
            data, y_column_name
        )

        plot_RR_interval(data,
                         x_column_name,
                         y_column_name,
                         anomalies=filtered_indices,
                         title=(
                             'RR-interval plot after removing '
                             f'of anomalies for {person}'
                         ),
                         saving_folder=saving_folder,
                         name=f'{person}_RR_intervals_with_anomalies')

        # Remove neighbouring heart beats to the selected ones
        data = remove_adjacent_beats(
            data,
            filtered_indices,
            parameters['adjacent_beats_for_removing']
        )

        # Remove day, month and year
        year = data.iloc[0]["Phone timestamp"].year
        month = data.iloc[0]["Phone timestamp"].month
        day = data.iloc[0]["Phone timestamp"].day
        data['Phone timestamp'] = pd.to_datetime(data['Phone timestamp']).dt.time

        # Load manually anomalies
        anomalies = load_arbitrary_dataframe(
            f'{main_folder}{person}/',
            f'anomalies_{person}.csv'
        )
        # Extend the range of each anomaly
        anomalies["Start"] = (
            pd.to_datetime(anomalies["Start"]) -
            pd.to_timedelta(parameters["adjacent_beats_for_removing"])
        ).dt.time
        anomalies["End"] = (
            pd.to_datetime(anomalies["End"]) +
            pd.to_timedelta(parameters["adjacent_beats_for_removing"])
        ).dt.time

        # pd.set_option('display.max_rows', None)
        # Filter out manually indicated anomalies
        masks = []
        for _, row in anomalies.iterrows():
            mask = ~data['Phone timestamp'].between(row['Start'], row['End'])
            masks.append(mask)
        combined_mask = pd.concat(masks, axis=1).all(axis=1)
        filtered_data = data[combined_mask]
        data = filtered_data.reset_index(drop=True)

        data['Phone timestamp'] = data['Phone timestamp'].apply(
            lambda x: datetime.datetime.combine(
                date(year, month, day), x)
        )
        # data["Phone timestamp"] = pd.Timestamp(data["Phone timestamp"])

        HRV_windows_values, median_timestamps = calculate_HRV_in_windows(
            data,
            step_frequency=parameters['step_frequency'],
            window_size=parameters['window_size'],
            method=parameters['method']
        )

        #################################################################
        # Make final plots
        plot_HRV_and_battles_results(
            HRV_windows_values,
            median_timestamps,
            main_folder,
            person,
            saving_folder,
            show_entering=True,
            method=parameters["method"],
            limited=False,
        )

        for battle in battles:
            plot_HRV_and_battles_results(
                HRV_windows_values,
                median_timestamps,
                main_folder,
                person,
                saving_folder,
                method=parameters["method"],
                limited=True,
                lower_bound=battle[0],
                upper_bound=battle[1]
            )
