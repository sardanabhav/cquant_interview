# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.preprocessing import MinMaxScaler

## TASK 1
df_16 = pd.read_csv('historicalPriceData\\ERCOT_DA_Prices_2016.csv', parse_dates=True, index_col='Date')
df_17 = pd.read_csv('historicalPriceData\\ERCOT_DA_Prices_2017.csv', parse_dates=True, index_col='Date')
df_18 = pd.read_csv('historicalPriceData\\ERCOT_DA_Prices_2018.csv', parse_dates=True, index_col='Date')
df_19 = pd.read_csv('historicalPriceData\\ERCOT_DA_Prices_2019.csv', parse_dates=True, index_col='Date')

combined_df = pd.concat([df_16, df_17, df_18, df_19], axis=0)


## TASK 2

combined_df['Year'] = combined_df.index.year
combined_df['Month'] = combined_df.index.month

group_df = combined_df.groupby(['SettlementPoint', 'Year', 'Month'])
average_prices_by_point_month = group_df.mean()
average_prices_by_point_month.columns = ['AveragePrice']

## TASK 3
avg_by_month = average_prices_by_point_month.reset_index()
avg_by_month.to_csv('output\\AveragePriceByMonth.csv', index=False)

## TASK 4
# remove negative and 0 prices
positive_prices = combined_df.loc[combined_df['Price'] > 0, ['SettlementPoint', 'Price']]

# Filter out hub data
positive_prices_hub = positive_prices[~positive_prices['SettlementPoint'].str.contains('LZ_')]
positive_prices_hub['logPrice'] = np.log(positive_prices_hub['Price'])

# hourly volatility
positive_prices_hub['Year'] = positive_prices_hub.index.year
#positive_prices_hub['month'] = positive_prices_hub.index.month
price_volatility = positive_prices_hub.groupby(['SettlementPoint', 'Year'])['logPrice'].std()

# write to csv
price_volatility = price_volatility.reset_index()
price_volatility.columns = ['SettlementPoint', 'Year', 'HourlyVolatility']
price_volatility.to_csv('output\\HourlyVolatilityByYear.csv', index=False)

## TASK 6
price_max_volatility = price_volatility.groupby('Year').max()
price_max_volatility = price_max_volatility.reset_index()
# reorder columns
price_max_volatility_rearranged = price_max_volatility.loc[:, ['SettlementPoint', 'Year', 'HourlyVolatility']]
# write to csv
price_max_volatility_rearranged.to_csv('output\\MaxVolatilityByYear.csv', index=False)


## TASK 7

for name, group in combined_df.groupby('SettlementPoint'):
    formatted_df = group.loc[:, ['SettlementPoint', 'Price']]
    formatted_df['Date'] = formatted_df.index.date
    formatted_df['Hour'] = formatted_df.index.hour
    formatted_df.index = [formatted_df['Date'], formatted_df['Hour']]
    formatted_df = formatted_df.loc[:, 'Price']
    formatted_df = formatted_df.reset_index(level=-1)
#    formatted_df['Hour'] = formatted_df['Hour'] + 1
#    formatted_df['Hour'] = 'X' + formatted_df['Hour'].astype(str)
    formatted_df = formatted_df.pivot(columns='Hour', values='Price')
    formatted_df.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6',
                            'X7', 'X8', 'X9', 'X10', 'X11', 'X12',
                            'X13', 'X14', 'X15', 'X16', 'X17', 'X18',
                            'X19', 'X20', 'X21', 'X22', 'X23', 'X24']
    formatted_df = formatted_df.reset_index()
    formatted_df.insert(0, 'Variable', name)
    formatted_df.to_csv('output\\formattedSpotHistory\\spot_' + name + '.csv', index=False)


## BONUS - Mean plots
avg_by_month_plot = avg_by_month.copy()
avg_by_month_plot.index = pd.to_datetime(avg_by_month['Year'].astype(str) + '/' +avg_by_month['Month'].astype(str) + '/01 00:00', format='%Y/%m/%d %H:%M')


# Plot average for settlement hubs

avg_by_month_hubs = avg_by_month_plot[~avg_by_month_plot['SettlementPoint'].str.contains('LZ_')]
avg_by_month_hubs['SettlementPoint'] = avg_by_month_hubs['SettlementPoint'].astype('category')
fig = plt.figure()
sns.lineplot(avg_by_month_hubs.index, avg_by_month_hubs['AveragePrice'], hue=avg_by_month_hubs['SettlementPoint'])
plt.xlabel('Time', fontsize = 20, fontweight='bold')
plt.ylabel('Average Price', fontsize = 20, fontweight='bold')
plt.xticks(size = 10)
plt.yticks(size = 20)
plt.legend(loc=2, prop={'size': 20, 'weight':'bold'},  mode="expand", borderaxespad=0., ncol=5)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.savefig('output\\SettlementHubAveragePriceByMonth.png', format='png', dpi=500, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)
# Plot average for load zones

avg_by_month_lz = avg_by_month_plot[avg_by_month_plot['SettlementPoint'].str.contains('LZ_')]
avg_by_month_lz['SettlementPoint'] = avg_by_month_lz['SettlementPoint'].astype('category')
fig = plt.figure()
sns.lineplot(avg_by_month_lz.index, avg_by_month_lz['AveragePrice'], hue=avg_by_month_lz['SettlementPoint'])
plt.xlabel('Time', fontsize = 20, fontweight='bold')
plt.ylabel('Average Price', fontsize = 20, fontweight='bold')
plt.xticks(size = 10)
plt.yticks(size = 20)
plt.legend(loc=2, prop={'size': 20, 'weight':'bold'},  mode="expand", borderaxespad=0., ncol=5)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.savefig('output\\LoadZoneAveragePriceByMonth.png', format='png', dpi=500, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)

## BONUS - Volatility plots
price_volatility_plot = price_volatility.copy()
price_volatility.index = pd.to_datetime(price_volatility['Year'].astype(str) + '/1/01 00:00', format='%Y/%m/%d %H:%M')


# Plot average for settlement hubs

price_volatility_hubs = price_volatility_plot[~price_volatility_plot['SettlementPoint'].str.contains('LZ_')]
price_volatility_hubs['Year'] = price_volatility_hubs['Year'].astype('category')
#price_volatility_hubs['SettlementPoint'] = price_volatility_hubs['SettlementPoint'].astype('category')
fig = plt.figure()
sns.barplot(price_volatility_hubs['SettlementPoint'], price_volatility_hubs['HourlyVolatility'], hue=price_volatility_hubs['Year'])
plt.xlabel('Settlement Hub', fontsize = 20, fontweight='bold')
plt.ylabel('Yearly Volatility', fontsize = 20, fontweight='bold')
plt.xticks(size = 10)
plt.yticks(size = 20)
plt.legend(loc=2, prop={'size': 20, 'weight':'bold'},  mode="expand", borderaxespad=0., ncol=5)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.savefig('output\\SettlementHubYearlyVolitility.png', format='png', dpi=500, bbox_inches='tight', pad_inches=0.1)
plt.close(fig)


## BONUS - Hourly Shape Profile

combined_df_bonus = combined_df.copy()
combined_df_bonus['day_of_week'] = combined_df_bonus.index.dayofweek
combined_df_bonus['month_of_year'] = combined_df_bonus.index.month
combined_df_bonus['hour_of_day'] = combined_df_bonus.index.hour

for name,group in combined_df_bonus.groupby('SettlementPoint'):
    hourly_shape_profile = group.groupby(['month_of_year', 
                                          'day_of_week', 
                                          'hour_of_day']).mean()['Price']
    hourly_shape_profile = pd.DataFrame(hourly_shape_profile)
    scalar = MinMaxScaler()
    scaled = scalar.fit_transform(hourly_shape_profile)
    hourly_shape_profile['NormalizedPrice'] = scaled
    hourly_shape_profile = hourly_shape_profile.reset_index()
    hourly_shape_profile.to_csv('output\\hourlyShapeProfiles\\profile_' + name + '.csv', index=False)













