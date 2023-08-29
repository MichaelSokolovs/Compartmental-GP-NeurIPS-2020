#!/usr/bin/env python
# coding: utf-8


import pickle
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pds
import torch
from pyro.ops.stats import quantile

import data_loader_res
import pyro_model.helper
import pyro_model.helper
from pyro_model.counterfactual_helper import get_R0_sooner_lockdown, get_R0_later_lockdown, get_counterfactual

countries = [
    'United Kingdom',
    'Italy',
    'Germany',
    'Spain',
    'US',
    'France',
    'Belgium',
    'Korea, South',
    'Brazil',
    'Iran',
    'Netherlands',
    'Canada',
    'Turkey',
    'Romania',
    'Portugal',
    'Sweden',
    'Switzerland',
    'Ireland',
    'Hungary',
    'Denmark',
    'Austria',
    'Mexico',
    'India',
    'Ecuador',
    'Russia',
    'Peru',
    'Indonesia',
    'Poland',
    'Philippines',
    'Japan',
    'Pakistan'
]

prefix = 'trained_models_custom/Loop14_custom_index_noC6wE/'
# prefix = ''

pad = 24

data_dict = data_loader_res.get_data_pyro(countries, smart_start=False, pad=pad, legacy=False)
print(data_dict['date_list'].get_loc('2020-04-20'))
print(data_dict['date_list'].shape)
print(data_dict['daily_death'].shape)
print(data_dict['daily_death'][103, 0])
data_dict = pyro_model.helper.smooth_daily(data_dict)

days = 14
train_len = data_dict['cum_death'].shape[0] - days
covariates_actual = pyro_model.helper.get_covariates_intervention(data_dict, train_len, notime=True)

def plot_graph(seed):
    model_id = 'day-{}-rng-{}'.format(days, seed)
    # = open(prefix + 'Loop{}/{}-predictive.pkl'.format(days, model_id), 'rb')
    f = open(prefix + '{}-predictive.pkl'.format(model_id), 'rb')
    res = pickle.load(f)
    f.close()
    f = open(prefix + '{}-forecaster.csv'.format(model_id), 'rb')
    #f = open(prefix + 'Loop{}/{}-forecaster.csv'.format(days, model_id), 'rb')
    forecaster = pickle.load(f)
    f.close()
    prediction = quantile(res['prediction'].squeeze(), (0.5,), dim=0).squeeze()

    start_ind = 14
    c = 0
    dt_list = data_dict['date_list']
    plt.plot(dt_list[:-start_ind - 1], np.diff(prediction[:, c]), label='Fitted SEIR')
    plt.plot(dt_list[:-start_ind - 1], data_dict['daily_death'][:-start_ind - 1, c], '.', label='acutal')

    plt.gcf().autofmt_xdate()
    plt.title(countries[c])
    plt.legend()
    R0 = res['R0'].squeeze()
    R0_counter7 = get_R0_sooner_lockdown(R0, 7)
    R0_counter7a = get_R0_later_lockdown(R0, 7)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.fill_between(dt_list[:-start_ind], quantile(R0, 0.025, dim=0)[0, :], quantile(R0, 0.975, dim=0)[0, :], color='orange',
                     alpha=0.3)

    plt.plot(dt_list[:-start_ind], torch.mean(R0, dim=0)[0, :])
    plt.axvline(datetime(2020, 3, 21), linestyle='--', color="black")
    plt.axvline(datetime(2020, 4, 20), linestyle='--', color="black")

    pred_counter7 = get_counterfactual(data_dict, forecaster, res, R0_counter7)
    pred_counter7_lo = pred_counter7[0]
    pred_counter7_up = pred_counter7[2]
    pred_counter7_me = pred_counter7[1]
    pred_counter7a = get_counterfactual(data_dict, forecaster, res, R0_counter7a)
    pred_counter7a_lo = pred_counter7a[0]
    pred_counter7a_up = pred_counter7a[2]
    pred_counter7a_me = pred_counter7a[1]
    pred_true = get_counterfactual(data_dict, forecaster, res, R0)[1, ...]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.fill_between(dt_list[:-start_ind-1], pred_counter7_lo[:, 0], pred_counter7_up[:, 0], color='blue', alpha=0.3)
    plt.fill_between(dt_list[:-start_ind-1], pred_counter7a_lo[:, 0], pred_counter7a_up[:, 0], color='orange', alpha=0.3)

    plt.plot(dt_list[:-start_ind-1], pred_counter7_me[:, 0], label='One week earlier', color='navy')
    plt.plot(dt_list[:-start_ind-1], pred_true[:, 0], label='Actual plan', color='black')
    plt.plot(dt_list[:-start_ind-1], pred_counter7a_me[:, 0], label='One week later', color='red')
    plt.plot(datetime(2020, 3, 24), 0, marker='x', markersize=5, color="black", label='Lockdown start dates',
             linestyle="None")
    plt.plot([datetime(2020, 3, 17)], [0], marker='x', markersize=5, color="navy")
    plt.plot([datetime(2020, 3, 31)], [0], marker='x', markersize=5, color="red")

    plt.legend()
    plt.savefig('tables/Fig-3-UK-counterfactual_custom_noC6_seed_{}.png'.format(seed), dpi=300)

    df_counterfactual = pds.DataFrame({
        'dt': dt_list[:-start_ind-1],
        'actual_plan': pred_true[:, 0],
        'wk_early_mean': pred_counter7_me[:, 0],
        'wk_early_lower': pred_counter7_lo[:, 0],
        'wk_early_upper': pred_counter7_up[:, 0],
        'wk_later_mean': pred_counter7a_me[:, 0],
        'wk_later_lower': pred_counter7a_lo[:, 0],
        'wk_later_upper': pred_counter7a_up[:, 0],
    })

    df_counterfactual.to_csv('tables/Fig-3-UK-counterfactual_custom_noC6_seed_{}.csv'.format(seed))
    plt.close()


for x in (0, 1, 3, 4, 5, 7, 8, 9, 10, 11):
    print("Processing seed " + str(x))
    plot_graph(x)
