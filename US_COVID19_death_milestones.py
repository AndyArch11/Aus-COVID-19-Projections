import decimal
import requests
import pandas as pd
import numpy as np
import math

from datetime import datetime
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#ECDC changed to weekly reporting, so Our World In Data changed to John Hopkins University to source their data, requiring change in path
#worldindata_covid_full_url = 'https://covid.ourworldindata.org/data/ecdc/full_data.csv'
worldindata_covid_full_url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
covid_data_filepath = './world_in_data_covid_full.csv'

US_2020_Pop = 330565500
year_start = datetime(2020, 1, 1)
date_first_covid_case = datetime.now()
date_first_covid_death = datetime.now()

def load_covid_deaths(download):
    if download:
        print('Retrieving file from Internet')
        #All datasources are problematic, currently using Our World in Data, but it could change
        response = requests.get(worldindata_covid_full_url, allow_redirects=True)
        if response.status_code == 200:
            print('File retrieved from Internet')
            try:
                #Save file for future reference, rather than download it directly into a panda dataframe
                open(covid_data_filepath, 'wb').write(response.content)
                print('File written to disk')
            except IOError:
                print('Error writing file to disk')
        else:
            print('Error retrieving file from World In Data: ' + response.status_code)
    
    print('Reading local file')
    return pd.read_csv(covid_data_filepath, usecols=['date', 'location', 'new_deaths', 'total_deaths', 'new_cases', 'total_cases'], parse_dates=['date'], index_col=['date'])

def read_US_mortality_stats():
    us_conflicts = pd.read_excel('./USA COVID-19 Infection Projections.xlsx', 'US Deaths', header=1, usecols=['Conflict', 'US Deaths', '% US Population', 'Daily Deaths'], nrows=18)
    us_leading_causes = pd.read_excel('./USA COVID-19 Infection Projections.xlsx', 'US Deaths', header=22, usecols=['Leading Causes', 'US Deaths', '% US Population', 'Daily Deaths'], nrows=11)
    us_epidemics = pd.read_excel('./USA COVID-19 Infection Projections.xlsx', 'US Deaths', header=36, usecols=['Epidemic', 'US Deaths', '% US Population', 'Daily Deaths'], nrows=11)
    us_disasters = pd.read_excel('./USA COVID-19 Infection Projections.xlsx', 'US Deaths', header=50, usecols=['US Disasters', 'US Deaths', '% US Population', 'Daily Deaths'], nrows=17)

    return us_conflicts, us_leading_causes, us_epidemics, us_disasters

def daily_covid_deaths_matplotlib(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    #print(plt.style.available)
    #['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']
    #plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(12, 12))
    ax2 = ax1.twinx()
    ax2.set_ylabel('Daily Confirmed Cases')
    ax2.set_yscale(scale)
    ax2.plot(usdf.index.values, usdf.new_cases, label='US Covid-19 Daily Confirmed Cases', color='0.6')
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    ax1.set(xlabel=xlabel, ylabel='Deaths', yscale=scale, title='US COVID-19 Reported Deaths per Day vs ' + title)
    ax1.bar(usdf.index.values, usdf.new_deaths, label='US Covid-19 Deaths reported on that day')
    date_format = ('%d %b')
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
    
    avg_deaths = usdf.loc[:,"new_deaths"].mean()
    avg_deaths_str = '{:,.1f}'.format(avg_deaths)
    if avg_deaths_str.endswith('.0'):
        avg_deaths_str = avg_deaths_str[:-2]
    ax1.axhline(y=avg_deaths, label='Average COVID-19 Deaths: ' + avg_deaths_str + ' deaths/day', linestyle='--', linewidth=1)

    add_mortality_lines(ax1, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit)    
    
    #Colormaps https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colormap = plt.get_cmap("gist_rainbow")
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])

    ax1.legend(loc="best")

    plt.show()

def percentage_covid_deaths_matplotlib(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    fig, ax1 = plt.subplots(figsize=(12, 12))
    ax2 = ax1.twinx()
    ax2.set_ylabel('Confirmed Cases as percentage of population')
    ax2.set_yscale(scale)
    ustotalpercdf = usdf['total_cases'].apply(lambda x: (x / US_2020_Pop) * 100).copy()
    ax2.plot(usdf.index.values, ustotalpercdf[0:], label='US Covid-19 Total Confirmed Cases: ' + '{:,.3f}'.format(ustotalpercdf.iloc[-1]) + '% of population', color='0.6')
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    ax1.set(xlabel=xlabel, ylabel='Deaths as percentage of population', yscale=scale, title='US COVID-19 Reported Deaths per Day as percentage of population vs ' + title)
    uspercdf = usdf['total_deaths'].apply(lambda x: (x / US_2020_Pop) * 100).copy()
    ax1.plot(usdf.index.values, uspercdf[0:], label='US Covid-19 Deaths: ' + '{:,.3f}'.format(uspercdf.iloc[-1]) + '% of population')
    date_format = ('%d %b')
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    
    usmortalitydf['perc_deaths'] = usmortalitydf[mortality_count_column].apply(lambda x: x * 100)
    add_mortality_lines(ax1, usdf, usmortalitydf, '{:,.3f}', 'perc_deaths', mortality_type_column, unit)    
    
    #Colormaps https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colormap = plt.get_cmap("gist_rainbow")
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])

    ax1.legend(loc="best")

    plt.show()

def total_covid_deaths_matplotlib(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    #plt.style.use('ggplot')
    fig, ax1 = plt.subplots(figsize=(12, 12))
    ax2 = ax1.twinx()
    ax2.set_ylabel('Confirmed Cases')
    ax2.set_yscale(scale)
    ax2.plot(usdf.index.values, usdf.total_cases, label='US Covid-19 Total Confirmed Cases: '  + str(usdf.total_cases.iloc[-1]), color='0.6')
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    ax1.set(xlabel=xlabel, ylabel='Deaths', yscale=scale, title='US COVID-19 Reported Cumulative Deaths vs ' + title)
    ax1.plot(usdf.index.values, usdf.total_deaths, label='US Covid-19 Deaths - total: ' + '{:,.0f}'.format(usdf['total_deaths'][-1]))
    date_format = ('%d %b')
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
    
    add_mortality_lines(ax1, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit)    
    
    #Colormaps https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colormap = plt.get_cmap("gist_rainbow")
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])

    ax1.legend(loc="best")

    plt.show()

def add_mortality_lines(ax, usdf, usmortalitydf, str_format, mortality_count_column, mortality_type_column, unit):
    sorted_mortality_df = usmortalitydf.sort_values(by=mortality_count_column, ascending=False)
    for index, row in sorted_mortality_df.iterrows():        
        total_deaths_str = str_format.format(row[mortality_count_column])
        if total_deaths_str.endswith('.0'):
            total_deaths_str = total_deaths_str[:-2]
        ax.axhline(y=row[mortality_count_column], label=(row[mortality_type_column] + ': ' + total_deaths_str + unit), linestyle='-', linewidth=1)

#______________________________________________________________________________________________________________________________

def daily_covid_deaths_seaborn(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    fig, ax1 = plt.subplots(figsize=(12, 12))
    ax2 = ax1.twinx()
    ax2.set_ylabel('Daily Confirmed Cases')
    ax2.set_yscale(scale)
    #ax2.plot(usdf.index.values, usdf.new_cases, label='US Covid-19 Daily Confirmed Cases', color='0.6')
    sns.lineplot(x=usdf.index.values, y=usdf.new_cases, ax=ax2)
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    ax1.set(xlabel=xlabel, ylabel='Deaths', yscale=scale, title='US COVID-19 Reported Deaths per Day vs ' + title)
    #ax1.bar(usdf.index.values, usdf.new_deaths, label='US Covid-19 Deaths reported on that day')
    sns.barplot(x=usdf.index.values, y=usdf.new_deaths, ax=ax1)
    date_format = ('%d %b')
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
    ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
    
    avg_deaths = usdf.loc[:,"new_deaths"].mean()
    avg_deaths_str = '{:,.1f}'.format(avg_deaths)
    if avg_deaths_str.endswith('.0'):
        avg_deaths_str = avg_deaths_str[:-2]
    ax1.axhline(y=avg_deaths, label='Average COVID-19 Deaths: ' + avg_deaths_str + ' deaths/day', linestyle='--', linewidth=1)

    add_mortality_lines(ax1, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit)    
    
    #Colormaps https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colormap = plt.get_cmap("gist_rainbow")
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
    for i,j in enumerate(ax1.lines):
        j.set_color(colors[i])

    ax1.legend(loc="best")

    plt.show()

#______________________________________________________________________________________________________________________________

def daily_covid_deaths_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit, filename):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=usdf.index, y=usdf.new_cases, name=('New Confirmed US Cases'), line=dict(width=1, dash='dot', color='rgb(100,100,100)')), secondary_y=True)
    fig.add_trace(go.Bar(x=usdf.index, y=usdf.new_deaths, name='US Covid-19 Deaths reported on that day'), secondary_y=False)

    avg_deaths = usdf.loc[:,"new_deaths"].mean()
    avg_deaths_str = '{:,.1f}'.format(avg_deaths)
    if avg_deaths_str.endswith('.0'):
        avg_deaths_str = avg_deaths_str[:-2]
    fig.add_trace(go.Scatter(x=usdf.index, y=np.linspace(avg_deaths, avg_deaths, len(usdf)), name=('Average COVID-19 Deaths: ' + avg_deaths_str + ' US deaths/day'), line=dict(width=1, dash='dot')), secondary_y=False)

    add_mortality_traces(fig, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit, True, False)    
    
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    fig.update_layout(title='US COVID-19 Reported Deaths per Day vs ' + title,
        xaxis_title=xlabel)
    # Set y-axes titles
    fig.update_yaxes(title_text="US Deaths", type=scale, separatethousands=True, secondary_y=False)
    fig.update_yaxes(title_text="Daily Confirmed US Cases", type=scale, separatethousands=True, secondary_y=True)
    
    fig.show(renderer = 'browser') #'json', 'png', 'jpeg', 'jpg', 'svg', 'pdf', 'browser', 'firefox', 'chrome', 'chromium'  

    export_image(fig, filename)  

def percentage_covid_deaths_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit, filename):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    ustotalpercdf = usdf['total_cases'].apply(lambda x: (x / US_2020_Pop)).copy()
    fig.add_trace(go.Scatter(x=usdf.index, y=ustotalpercdf[0:], name=('Total Confirmed US Cases: ' + '{:,.3f}'.format(ustotalpercdf.iloc[-1] * 100) + '% of US population'), line=dict(width=1, dash='dot', color='rgb(100,100,100)')), secondary_y=True)
    uspercdf = usdf['total_deaths'].apply(lambda x: (x / US_2020_Pop)).copy()
    fig.add_trace(go.Scatter(x=usdf.index, y=uspercdf[0:], name='US Covid-19 Deaths: ' + '{:,.3f}'.format(uspercdf.iloc[-1] * 100) + '% of US population'), secondary_y=False)
    
    #usmortalitydf['perc_deaths'] = usmortalitydf[mortality_count_column].apply(lambda x: x * 100)
    add_mortality_traces(fig, usdf, usmortalitydf, '{:,.3f}', mortality_count_column, mortality_type_column, unit, True, True)    

    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    fig.update_layout(title='US COVID-19 Reported Deaths per Day as percentage of US population vs ' + title,
        xaxis_title=xlabel)
    # Set y-axes titles
    fig.update_yaxes(title_text='US Deaths as percentage of US population', type=scale, separatethousands=True, secondary_y=False, tickformat=',.5%')
    fig.update_yaxes(title_text="Daily Confirmed US Cases as percentage of US population", type=scale, separatethousands=True, secondary_y=True, tickformat=',.5%')
    
    fig.show(renderer = 'browser')

    export_image(fig, filename)

def percentage_covid_deaths_horiz_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit, filename, tick_format):
    #'Conflict', 'US Deaths', '% US Population', 'Daily Deaths'
    covid = pd.Series({usmortalitydf.columns[0]: 'COVID-19 US Deaths', usmortalitydf.columns[1]: usdf.total_deaths[-1], usmortalitydf.columns[2]: (usdf.total_deaths[-1] / US_2020_Pop), usmortalitydf.columns[3]: usdf.loc[:,"new_deaths"].mean()})
    #usmortalitydf['perc_deaths'] = usmortalitydf[mortality_count_column].apply(lambda x: x * 100)
    usmortalitydf = usmortalitydf.append(covid, ignore_index = True, sort=False)
    sorted_mortality_df = usmortalitydf.sort_values(by=mortality_count_column, ascending=True)
    sorted_mortality_df.reset_index(inplace=True, drop=True)
    colours = ['lightslategrey',] * len(sorted_mortality_df)
    idx = sorted_mortality_df.loc[sorted_mortality_df[usmortalitydf.columns[0]]=='COVID-19 US Deaths']
    colours[idx.index.values[0]] = 'crimson'

    fig = go.Figure(go.Bar(x=sorted_mortality_df['% US Population'], y=sorted_mortality_df[usmortalitydf.columns[0]], orientation='h', marker_color=colours))
    fig.update_xaxes(type=scale)

    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]    
    fig.update_xaxes(title_text='US Deaths as percentage of US population', type=scale, separatethousands=True, tickformat=tick_format)
    fig.update_layout(title='US COVID-19 Reported Deaths per Day as percentage of US population vs ' + title, xaxis_title=xlabel)
    
    fig.show(renderer = 'browser') #'json', 'png', 'jpeg', 'jpg', 'svg', 'pdf', 'browser', 'firefox', 'chrome', 'chromium' 

    export_image(fig, filename)


def total_covid_deaths_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit, filename):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=usdf.index, y=usdf.total_cases, name=('Total Confirmed US Cases: ' + '{:,.0f}'.format(usdf['total_cases'][-1])), line=dict(width=1, dash='dot', color='rgb(100,100,100)')), secondary_y=True)
    fig.add_trace(go.Scatter(x=usdf.index, y=usdf.total_deaths, name='US Covid-19 Deaths - total: ' + '{:,.0f}'.format(usdf['total_deaths'][-1])), secondary_y=False)

    add_mortality_traces(fig, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit, True, False)    

    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    fig.update_layout(title='US COVID-19 Reported Cumulative US Deaths vs ' + title,
        xaxis_title=xlabel)
    # Set y-axes titles
    fig.update_yaxes(title_text='US Deaths', type=scale, separatethousands=True, secondary_y=False)
    fig.update_yaxes(title_text="Confirmed US Cases", type=scale, separatethousands=True, secondary_y=True)
    
    fig.show(renderer = 'browser') #'json', 'png', 'jpeg', 'jpg', 'svg', 'pdf', 'browser', 'firefox', 'chrome', 'chromium' 

    export_image(fig, filename)

def total_covid_deaths_horiz_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit, filename):
    #'Conflict', 'US Deaths', '% US Population', 'Daily Deaths'
    covid = pd.Series({usmortalitydf.columns[0]: 'COVID-19 US Deaths', usmortalitydf.columns[1]: usdf.total_deaths[-1], usmortalitydf.columns[2]: (usdf.total_deaths[-1] / US_2020_Pop) * 100, usmortalitydf.columns[3]: usdf.loc[:,'new_deaths'].mean()})
    mortalitydf = usmortalitydf
    mortalitydf = mortalitydf.append(covid, ignore_index = True, sort=False)
    sorted_mortality_df = mortalitydf.sort_values(by=mortality_count_column, ascending=True)
    sorted_mortality_df.reset_index(inplace=True, drop=True)
    colours = ['lightslategrey',] * len(sorted_mortality_df)
    idx = sorted_mortality_df.loc[sorted_mortality_df[usmortalitydf.columns[0]]=='COVID-19 US Deaths']
    colours[idx.index.values[0]] = 'crimson'

    fig = go.Figure(go.Bar(x=sorted_mortality_df['US Deaths'], y=sorted_mortality_df[usmortalitydf.columns[0]], orientation='h', marker_color=colours))
    fig.update_xaxes(type=scale)

    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]    
    fig.update_xaxes(title_text='US Deaths', type=scale, separatethousands=True, tickformat=',.0f')
    fig.update_layout(title='US COVID-19 Reported Cumulative US Deaths vs ' + title, xaxis_title=xlabel)
    
    fig.show(renderer = 'browser') #'json', 'png', 'jpeg', 'jpg', 'svg', 'pdf', 'browser', 'firefox', 'chrome', 'chromium' 

    export_image(fig, filename)

def animated_total_covid_deaths_horiz_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit, date_from):
    number_frames = (usdf.index[-1] - date_from).days + 1
    frame_duration = 300
    frame_transition_duration = 200

    us_sorted_mortality = usmortalitydf.sort_values(by=mortality_count_column, ascending=True)
    init_x_data = us_sorted_mortality[mortality_count_column]
    init_x_data = pd.concat([pd.Series([0.0]), init_x_data], axis=0, ignore_index=True)
    init_y_data = us_sorted_mortality[mortality_type_column]
    init_y_data = pd.concat([pd.Series(['COVID-19 US Deaths']), init_y_data], axis=0, ignore_index=True)

    covid_progress = []
    steps = []

    for day in range(number_frames):
        current_day = usmortalitydf.copy()
        current_day['Daily Deaths'] = usmortalitydf['Daily Deaths'].apply(lambda x: (x * (day + 1))).copy()
        current_year = date_from + timedelta(days=day)
        current_covid_deaths = usdf[current_year:current_year].total_deaths[0]
        if math.isnan(current_covid_deaths):
            current_covid_deaths = 0
        #'Conflict|Leading Causes|Epidemics', 'US Deaths', '% US Population', 'Daily Deaths'
        current_day = current_day.append(pd.Series({usmortalitydf.columns[0]: 'COVID-19 US Deaths', usmortalitydf.columns[1]: current_covid_deaths, usmortalitydf.columns[2]: (current_covid_deaths / US_2020_Pop) * 100, usmortalitydf.columns[3]: (current_covid_deaths)}), ignore_index=True)
        sorted_current_day = current_day.sort_values(by=mortality_count_column, ascending=True).reset_index(drop=True)        
        colours = ['lightslategrey',] * len(sorted_current_day)
        idx = sorted_current_day.loc[sorted_current_day[sorted_current_day.columns[0]]=='COVID-19 US Deaths']
        colours[idx.index.values[0]] = 'crimson'
        sorted_current_day['colours'] = colours
        covid_progress.append(sorted_current_day)

        step = dict(method='animate', args=[[day], dict(mode='immediate', transition=dict(duration=frame_transition_duration), frame=dict(duration=frame_duration, redraw=True))], label=current_year.strftime('%d/%m/%y'))
        steps.append(step)

    upper_range = covid_progress[-1][mortality_count_column].nlargest(1)[len(covid_progress[-1][mortality_count_column])-1]
    if scale == 'log':
        upper_range = math.log10(upper_range)
    upper_range *= 1.01

    fig = go.Figure(
        data = [go.Bar(x=init_x_data, y=init_y_data, orientation='h', marker_color=covid_progress[0]['colours'])],
        layout=go.Layout(
            xaxis=dict(range=[0, upper_range], autorange=False, title='US Deaths', type=scale),
            yaxis=dict(range=[covid_progress[-1][mortality_type_column], covid_progress[-1][mortality_type_column]], title=mortality_type_column),
            title=title + ' - Day: ' + str(1) + ', ' + (date_from).strftime('%d/%m/%Y') + '. US Covid Cases: ' + '{:,.0f}'.format(int(usdf.loc[date_from:date_from]['total_cases'])) + '. US Covid Deaths: ' + '{:,.0f}'.format(int(covid_progress[0].loc[(covid_progress[0][mortality_type_column] == 'COVID-19 US Deaths')][mortality_count_column])),
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play", 
                                method="animate", 
                                args=[None, dict(frame=dict(duration=frame_duration, redraw=True), fromcurrent=True, transition=dict(duration=frame_transition_duration, easing='quadratic-in-out'))]),
                        dict(label="Pause", 
                                method="animate", 
                                args=[[None], dict(frame=dict(duration=0, redraw=True), mode='immediate', transition=dict(duration=0))])],
                direction='left',
                showactive=False,
                xanchor='right',
                yanchor='top',
                pad=dict(r=10, t=180),
                x=0.1,
                y=0)],
            sliders=[dict(
                active=0, 
                yanchor='top', 
                xanchor='left', 
                currentvalue=dict(
                    font=dict(size = 20), 
                    prefix='Date: ', 
                    visible=True, 
                    xanchor='right'),
                transition=dict(duration=frame_transition_duration, easing='cubic-in-out'), 
                pad=dict(b=10, t=50), 
                len=0.9, 
                x=0.1, 
                y=0, 
                steps=steps)]
        ),
        frames=[go.Frame(
            name=day_frame,
            data=[go.Bar(
                x=covid_progress[day_frame][mortality_count_column],
                y=covid_progress[day_frame][mortality_type_column],
                orientation='h',
                marker_color=covid_progress[day_frame]['colours'])],
            layout=go.Layout(
                title_text=title + ' - Day: ' + str(day_frame + 1) + ', ' + (date_from + timedelta(days=day_frame)).strftime('%d/%m/%Y') + '. US Covid Cases: ' + '{:,.0f}'.format(int(usdf[(date_from + timedelta(days=day_frame)):(date_from + timedelta(days=day_frame))]['total_cases'])) + '. US Covid Deaths: ' + '{:,.0f}'.format(int(covid_progress[day_frame].loc[(covid_progress[day_frame][mortality_type_column] == 'COVID-19 US Deaths')][mortality_count_column]))
                ))
            for day_frame in range(number_frames)]
    )
    
    fig.show()



def add_mortality_traces(fig, usdf, usmortalitydf, str_format, mortality_count_column, mortality_type_column, unit, has_secondary_y, is_perc):
    usmortalitydf[mortality_count_column] = usmortalitydf[mortality_count_column].fillna(0)
    sorted_mortality_df = usmortalitydf.sort_values(by=mortality_count_column, ascending=False)
    for index, row in sorted_mortality_df.iterrows():
        total_deaths = row[mortality_count_column]
        if is_perc:
            total_deaths *= 100
        total_deaths_str = str_format.format(total_deaths)
        if total_deaths_str.endswith('.0'):
            total_deaths_str = total_deaths_str[:-2]
        if has_secondary_y:
            fig.add_trace(go.Scatter(x=usdf.index, y=np.linspace(row[mortality_count_column], row[mortality_count_column], len(usdf)), name=(row[mortality_type_column] + ': ' + total_deaths_str + unit), line = dict(width=1)), secondary_y=False)
        else:
            fig.add_trace(go.Scatter(x=usdf.index, y=np.linspace(row[mortality_count_column], row[mortality_count_column], len(usdf)), name=(row[mortality_type_column] + ': ' + total_deaths_str + unit), line = dict(width=1)))

def export_image(fig, filename):
    fig.write_image('./Charts/' + filename + '.png', width=2500, height=1400)

def main():
    coviddf = load_covid_deaths(True)
    usdf_full = coviddf[(coviddf.location == 'United States')].fillna(0)
    usdf_cases = coviddf[(coviddf.location == 'United States') & (coviddf.total_cases > 0)].fillna(0)
    date_first_covid_case = usdf_cases.first_valid_index()
    usdf = coviddf[(coviddf.location == 'United States') & (coviddf.total_deaths > 0)].fillna(0)
    date_first_covid_death = usdf.first_valid_index()

    us_conflicts_df, us_leading_causes_df, us_epidemics_df, us_disasters_df = read_US_mortality_stats()

    #Matplotlib Charts
    #daily_covid_deaths_matplotlib('US Conflict Daily US Death Rates', usdf, 'linear', us_conflicts_df, 'Daily Deaths', 'Conflict', ' US deaths/day')
    #percentage_covid_deaths_matplotlib('US Conflicts as percentage of US population', usdf, 'log', us_conflicts_df, '% US Population', 'Conflict', '% of US population')
    #total_covid_deaths_matplotlib('US Conflicts', usdf, 'log', us_conflicts_df, 'US Deaths', 'Conflict', ' US deaths')
    #daily_covid_deaths_matplotlib('US Leading Causes of Death of 2019, Daily US Death Rates', usdf, 'linear', us_leading_causes_df, 'Daily Deaths', 'Leading Causes', ' US deaths/day')
    #percentage_covid_deaths_matplotlib('US Leading Causes of Death of 2019 as percentage of US population', usdf, 'log', us_leading_causes_df, '% US Population', 'Leading Causes', '% of US population')
    #total_covid_deaths_matplotlib('US Leading Causes of Death of 2019', usdf, 'log', us_leading_causes_df, 'US Deaths', 'Leading Causes', ' US deaths')
    #daily_covid_deaths_matplotlib('US Epidemics, Daily US Death Rates', usdf, 'linear', us_epidemics_df, 'Daily Deaths', 'Epidemic', ' US deaths/day')
    #percentage_covid_deaths_matplotlib('US Epidemics as percentage of US population', usdf, 'log', us_epidemics_df, '% US Population', 'Epidemic', '% of US population')
    #total_covid_deaths_matplotlib('US Epidemics', usdf, 'log', us_epidemics_df, 'US Deaths', 'Epidemic', ' US deaths')
    #daily_covid_deaths_matplotlib('US Disasters', usdf, 'linear', us_disasters_df, 'US Deaths', 'US Disasters', ' US deaths')
    #percentage_covid_deaths_matplotlib('US Disasters as percentage of US population', usdf, 'log', us_disasters_df, '% US Population', 'US Disasters', '% of US population')
    #total_covid_deaths_matplotlib('US Disasters', usdf, 'log', us_disasters_df, 'US Deaths', 'US Disasters', ' US deaths')

    #_______________________________________________________________________________________________________________________________________________

    #seaborn
    #daily_covid_deaths_seaborn('US Conflict Daily Death Rates', usdf, 'linear', us_conflicts_df, 'Daily Deaths', 'Conflict', ' US deaths/day')

    #_______________________________________________________________________________________________________________________________________________

    #Animated horizontal Plotly Charts
    
    # data set no longer begins at the start of the year - the following three no longer work.
    #animated_total_covid_deaths_horiz_plotly('US Conflicts', usdf_full, 'log', us_conflicts_df, 'US Deaths', 'Conflict', ' US deaths', year_start)
    #animated_total_covid_deaths_horiz_plotly('US Leading Causes of Death of 2019', usdf_full, 'log', us_leading_causes_df, 'US Deaths', 'Leading Causes', ' US deaths', year_start)
    #animated_total_covid_deaths_horiz_plotly('US Epidemics', usdf_full, 'log', us_epidemics_df, 'US Deaths', 'Epidemic', ' US deaths', year_start)

    #animated_total_covid_deaths_horiz_plotly('US Conflicts', usdf_cases, 'log', us_conflicts_df, 'US Deaths', 'Conflict', ' US deaths', date_first_covid_case)
    #animated_total_covid_deaths_horiz_plotly('US Leading Causes of Death of 2019', usdf_cases, 'log', us_leading_causes_df, 'Daily Deaths', 'Leading Causes', ' US deaths', date_first_covid_case)
    #animated_total_covid_deaths_horiz_plotly('US Epidemics', usdf_cases, 'log', us_epidemics_df, 'Daily Deaths', 'Epidemic', ' US deaths', date_first_covid_case)

    #animated_total_covid_deaths_horiz_plotly('US Conflicts', usdf, 'log', us_conflicts_df, 'US Deaths', 'Conflict', ' US deaths', date_first_covid_death)
    #animated_total_covid_deaths_horiz_plotly('US Leading Causes of Death of 2019', usdf, 'log', us_leading_causes_df, 'Daily Deaths', 'Leading Causes', ' US deaths', date_first_covid_death)
    #animated_total_covid_deaths_horiz_plotly('US Epidemics', usdf, 'log', us_epidemics_df, 'Daily Deaths', 'Epidemic', ' US deaths', date_first_covid_death)
    #exit()

    #_______________________________________________________________________________________________________________________________________________

    #Plotly Charts
    daily_covid_deaths_plotly('US Conflicts, Daily US Death Rates', usdf, 'linear', us_conflicts_df, 'Daily Deaths', 'Conflict', ' US deaths/day', 'US Conflicts Daily')
    percentage_covid_deaths_plotly('US Conflicts as percentage of US population', usdf, 'log', us_conflicts_df, '% US Population', 'Conflict', '% of US population', 'US Conflicts Pop Percentage')
    percentage_covid_deaths_horiz_plotly('US Conflicts as percentage of US population', usdf, 'log', us_conflicts_df, '% US Population', 'Conflict', '% of US population', 'US Conflicts Pop Percentage - horizbar', ',.4%')
    total_covid_deaths_plotly('US Conflicts', usdf, 'log', us_conflicts_df, 'US Deaths', 'Conflict', ' US deaths', 'US Conflicts')
    total_covid_deaths_horiz_plotly('US Conflicts', usdf, 'log', us_conflicts_df, 'US Deaths', 'Conflict', ' US deaths', 'US Conflicts - horizbar')
    daily_covid_deaths_plotly('US Leading Causes of Death of 2019, Daily US Death Rates', usdf, 'linear', us_leading_causes_df, 'Daily Deaths', 'Leading Causes', ' US deaths/day', 'US Leading Causes Daily')
    percentage_covid_deaths_plotly('US Leading Causes of Death of 2019 as percentage of US population', usdf, 'log', us_leading_causes_df, '% US Population', 'Leading Causes', '% of US population', 'US Leading Causes Pop Percentage')
    percentage_covid_deaths_horiz_plotly('US Leading Causes of Death of 2019 as percentage of US population', usdf, 'log', us_leading_causes_df, '% US Population', 'Leading Causes', '% of US population', 'US Leading Causes Pop Percentage - horizbar', ',.2%')
    total_covid_deaths_plotly('US Leading Causes of Death of 2019', usdf, 'log', us_leading_causes_df, 'US Deaths', 'Leading Causes', ' US deaths', 'US Leading Causes')
    total_covid_deaths_horiz_plotly('US Leading Causes of Death of 2019', usdf, 'log', us_leading_causes_df, 'US Deaths', 'Leading Causes', ' US deaths', 'US Leading Causes - horizbar')
    daily_covid_deaths_plotly('US Epidemics, Daily US Death Rates', usdf, 'linear', us_epidemics_df, 'Daily Deaths', 'Epidemic', ' US deaths/day', 'US Epidemics Daily')
    percentage_covid_deaths_plotly('US Epidemics as percentage of US population', usdf, 'log', us_epidemics_df, '% US Population', 'Epidemic', '% of US population', 'US Epidemics Pop Percentage')
    percentage_covid_deaths_horiz_plotly('US Epidemics as percentage of US population', usdf, 'log', us_epidemics_df, '% US Population', 'Epidemic', '% of US population', 'US Epidemics Pop Percentage - horizbar', ',.2%')
    total_covid_deaths_plotly('US Epidemics', usdf, 'log', us_epidemics_df, 'US Deaths', 'Epidemic', ' US deaths', 'US Epidemics')
    total_covid_deaths_horiz_plotly('US Epidemics', usdf, 'log', us_epidemics_df, 'US Deaths', 'Epidemic', ' US deaths', 'US Epidemics - horizbar')
    daily_covid_deaths_plotly('US Disasters', usdf, 'linear', us_disasters_df, 'US Deaths', 'US Disasters', ' US deaths', 'US Disasters Daily')
    percentage_covid_deaths_plotly('US Disasters as percentage of US population', usdf, 'log', us_disasters_df, '% US Population', 'US Disasters', '% of US population', 'US Disasters Pop Percentage')
    percentage_covid_deaths_horiz_plotly('US Disasters as percentage of US population', usdf, 'log', us_disasters_df, '% US Population', 'US Disasters', '% of US population', 'US Disasters Pop Percentage - horizbar', ',.3%')
    total_covid_deaths_plotly('US Disasters', usdf, 'log', us_disasters_df, 'US Deaths', 'US Disasters', ' US deaths', 'US Disasters')
    total_covid_deaths_horiz_plotly('US Disasters', usdf, 'log', us_disasters_df, 'US Deaths', 'US Disasters', ' US deaths', 'US Disasters - horizbar')

main()

