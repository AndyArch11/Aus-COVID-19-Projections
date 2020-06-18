import decimal
import requests
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# Handle date time conversions between pandas and matplotlib
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

worldindata_covid_full_url = 'https://covid.ourworldindata.org/data/ecdc/full_data.csv'
covid_data_filepath = './world_in_data_covid_full.csv'

def load_covid_deaths(download):
    if download:
        print('Retrieving file from Internet')
        #All datasources are pooblematic, currently using Our World in Data, but it could change
        response = requests.get(worldindata_covid_full_url, allow_redirects=True)
        if response.status_code == 200:
            try:
                #Save file for future reference, rather than download it directly into a panda dataframe
                open(covid_data_filepath, 'wb').write(response.content)
            except IOError:
                print('Error writing file to disk')
            finally:
                print('File written to disk')
        else:
            print('Error retrieving file from World In Data: ' + response.status_code)
    
    print('Reading local file')
    return pd.read_csv(covid_data_filepath, usecols=['date', 'location', 'new_deaths', 'total_deaths'], parse_dates=['date'], index_col=['date']) #dropping 'new_cases' and 'total_cases' columns

def read_US_mortality_stats():
    us_conflicts = pd.read_excel('./USA COVID-19 Infection Projections.xlsx', 'US Deaths', header=1, usecols=['Conflict', 'US Deaths', '% US Population', 'Daily Deaths'], nrows=18)
    us_leading_causes = pd.read_excel('./USA COVID-19 Infection Projections.xlsx', 'US Deaths', header=22, usecols=['Leading Causes', 'US Deaths', '% US Population', 'Daily Deaths'], nrows=11)
    us_epidemics = pd.read_excel('./USA COVID-19 Infection Projections.xlsx', 'US Deaths', header=36, usecols=['Epidemic', 'US Deaths', '% US Population', 'Daily Deaths'], nrows=11)
    us_disasters = pd.read_excel('./USA COVID-19 Infection Projections.xlsx', 'US Deaths', header=50, usecols=['US Disasters', 'US Deaths', '% US Population'], nrows=17)

    return us_conflicts, us_leading_causes, us_epidemics, us_disasters

def daily_covid_deaths_matplotlib(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    #print(plt.style.available)
    #['bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn', 'Solarize_Light2', 'tableau-colorblind10', '_classic_test']
    #plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 12))
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    ax.set(xlabel=xlabel, ylabel='Deaths', yscale=scale, title='US COVID-19 Reported Deaths per Day vs ' + title)
    ax.bar(usdf.index.values, usdf.new_deaths, label='US Covid-19 Deaths reported on that day')
    date_format = ('%d %b')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
    
    avg_deaths = usdf.mean(axis=0)
    avg_deaths_str = '{:,.1f}'.format(avg_deaths[0])
    if avg_deaths_str.endswith('.0'):
        avg_deaths_str = avg_deaths_str[:-2]
    ax.axhline(y=avg_deaths[0], label='Average COVID-19 Deaths: ' + avg_deaths_str + ' deaths/day', linestyle='--', linewidth=1)

    add_mortality_lines(ax, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit)    
    
    #Colormaps https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colormap = plt.get_cmap("gist_rainbow")
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(loc="best")

    plt.show()

def percentage_covid_deaths_matplotlib(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    fig, ax = plt.subplots(figsize=(12, 12))
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    ax.set(xlabel=xlabel, ylabel='Deaths as percentage of population', yscale=scale, title='US COVID-19 Reported Deaths per Day as percentage of population vs ' + title)
    uspercdf = usdf['total_deaths'].apply(lambda x: (x / 330565500) * 100).copy()
    print(uspercdf)
    ax.plot(usdf.index.values, uspercdf[0:], label='US Covid-19 Deaths as percentage of population')
    date_format = ('%d %b')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    
    usmortalitydf['perc_deaths'] = usmortalitydf[mortality_count_column].apply(lambda x: x * 100)
    add_mortality_lines(ax, usdf, usmortalitydf, '{:,.3f}', 'perc_deaths', mortality_type_column, unit)    
    
    #Colormaps https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colormap = plt.get_cmap("gist_rainbow")
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(loc="best")

    plt.show()

def total_covid_deaths_matplotlib(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    #plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 12))
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    ax.set(xlabel=xlabel, ylabel='Deaths', yscale=scale, title='US COVID-19 Reported Cumulative Deaths vs ' + title)
    ax.plot(usdf.index.values, usdf.total_deaths, label='US Covid-19 Deaths reported on that day')
    date_format = ('%d %b')
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda y, p: format(int(y), ',')))
    
    add_mortality_lines(ax, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit)    
    
    #Colormaps https://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colormap = plt.get_cmap("gist_rainbow")
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax.lines))]
    for i,j in enumerate(ax.lines):
        j.set_color(colors[i])

    ax.legend(loc="best")

    plt.show()

def add_mortality_lines(ax, usdf, usmortalitydf, str_format, mortality_count_column, mortality_type_column, unit):
    sorted_mortality_df = usmortalitydf.sort_values(by=mortality_count_column, ascending=False)
    for index, row in sorted_mortality_df.iterrows():        
        total_deaths_str = str_format.format(row[mortality_count_column])
        if total_deaths_str.endswith('.0'):
            total_deaths_str = total_deaths_str[:-2]
        ax.axhline(y=row[mortality_count_column], label=(row[mortality_type_column] + ': ' + total_deaths_str + unit), linestyle='-', linewidth=1)


def daily_covid_deaths_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    fig = go.Figure(layout_title_text='US COVID-19 Reported Deaths per Day vs ' + title)
    fig.add_trace(go.Bar(x=usdf.index, y=usdf.new_deaths, name='US Covid-19 Deaths reported on that day'))

    avg_deaths = usdf.mean(axis=0)
    avg_deaths_str = '{:,.1f}'.format(avg_deaths[0])
    if avg_deaths_str.endswith('.0'):
        avg_deaths_str = avg_deaths_str[:-2]
    fig.add_trace(go.Scatter(x=usdf.index, y=np.linspace(avg_deaths[0], avg_deaths[0], len(usdf)), name=('Average COVID-19 Deaths: ' + avg_deaths_str + ' deaths/day'), line = dict(width=1, dash='dot')))

    add_mortality_traces(fig, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit)    
    
    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    fig.update_layout(yaxis_type=scale, xaxis_title=xlabel, yaxis_title='Deaths')
    
    fig.show(renderer = 'browser') #'json', 'png', 'jpeg', 'jpg', 'svg', 'pdf', 'browser', 'firefox', 'chrome', 'chromium'    

def percentage_covid_deaths_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    fig = go.Figure(layout_title_text='US COVID-19 Reported Deaths per Day as percentage of population vs ' + title)
    uspercdf = usdf['total_deaths'].apply(lambda x: (x / 330565500) * 100).copy()
    fig.add_trace(go.Scatter(x=usdf.index, y=uspercdf[0:], name='US Covid-19 Deaths as percentage of population'))
    
    usmortalitydf['perc_deaths'] = usmortalitydf[mortality_count_column].apply(lambda x: x * 100)
    add_mortality_traces(fig, usdf, usmortalitydf, '{:,.3f}', 'perc_deaths', mortality_type_column, unit)    

    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    fig.update_layout(yaxis_type=scale, xaxis_title=xlabel, yaxis_title='Deaths as percentage of population')
    
    fig.show(renderer = 'browser')


def total_covid_deaths_plotly(title, usdf, scale, usmortalitydf, mortality_count_column, mortality_type_column, unit):
    fig = go.Figure(layout_title_text='US COVID-19 Reported Cumulative Deaths vs ' + title)
    fig.add_trace(go.Scatter(x=usdf.index, y=usdf.total_deaths, name='US Covid-19 Deaths reported on that day'))

    add_mortality_traces(fig, usdf, usmortalitydf, '{:,.1f}', mortality_count_column, mortality_type_column, unit)    

    xlabel = 'Date: ' + usdf.index[0].strftime('%d/%m/%Y') + ' - ' + usdf.index[-1].strftime('%d/%m/%Y') + ', ' + str(usdf.index[-1] - usdf.index[0])[:-9]
    fig.update_layout(yaxis_type=scale, xaxis_title=xlabel, yaxis_title='Deaths')
    
    fig.show(renderer = 'browser') #'json', 'png', 'jpeg', 'jpg', 'svg', 'pdf', 'browser', 'firefox', 'chrome', 'chromium' 

def add_mortality_traces(fig, usdf, usmortalitydf, str_format, mortality_count_column, mortality_type_column, unit):
    sorted_mortality_df = usmortalitydf.sort_values(by=mortality_count_column, ascending=False)
    for index, row in sorted_mortality_df.iterrows():
        total_deaths_str = str_format.format(row[mortality_count_column])
        if total_deaths_str.endswith('.0'):
            total_deaths_str = total_deaths_str[:-2]
        fig.add_trace(go.Scatter(x=usdf.index, y=np.linspace(row[mortality_count_column], row[mortality_count_column], len(usdf)), name=(row[mortality_type_column] + ': ' + total_deaths_str + unit), line = dict(width=1)))

def main():
    coviddf = load_covid_deaths(True)
    usdf = coviddf[(coviddf.location == 'United States') & (coviddf.total_deaths > 0)]

    us_conflicts_df, us_leading_causes_df, us_epidemics_df, us_disasters_df = read_US_mortality_stats()

    #Matplotlib Charts
    #daily_covid_deaths_matplotlib('US Conflict Daily Death Rates', usdf, 'linear', us_conflicts_df, 'Daily Deaths', 'Conflict', ' deaths/day')
    #percentage_covid_deaths_matplotlib('US Conflicts as percentage of population', usdf, 'log', us_conflicts_df, '% US Population', 'Conflict', '% of population')
    #total_covid_deaths_matplotlib('US Conflicts', usdf, 'log', us_conflicts_df, 'US Deaths', 'Conflict', ' deaths')
    #daily_covid_deaths_matplotlib('US Leading Causes of Death, Daily Death Rates', usdf, 'linear', us_leading_causes_df, 'Daily Deaths', 'Leading Causes', ' deaths/day')
    #percentage_covid_deaths_matplotlib('US Leading Causes of Death as percentage of population', usdf, 'log', us_leading_causes_df, '% US Population', 'Leading Causes', '% of population')
    #total_covid_deaths_matplotlib('US Leading Causes of Death', usdf, 'log', us_leading_causes_df, 'US Deaths', 'Leading Causes', ' deaths')
    #daily_covid_deaths_matplotlib('US Epidemics, Daily Death Rates', usdf, 'linear', us_epidemics_df, 'Daily Deaths', 'Epidemic', ' deaths/day')
    #percentage_covid_deaths_matplotlib('US Epidemics as percentage of population', usdf, 'log', us_epidemics_df, '% US Population', 'Epidemic', '% of population')
    #total_covid_deaths_matplotlib('US Epidemics', usdf, 'log', us_epidemics_df, 'US Deaths', 'Epidemic', ' deaths')
    #daily_covid_deaths_matplotlib('US Disasters', usdf, 'linear', us_disasters_df, 'US Deaths', 'US Disasters', ' deaths')
    #percentage_covid_deaths_matplotlib('US Disasters as percentage of population', usdf, 'log', us_disasters_df, '% US Population', 'US Disasters', '% of population')
    #total_covid_deaths_matplotlib('US Disasters', usdf, 'log', us_disasters_df, 'US Deaths', 'US Disasters', ' deaths')

    #Plotly Charts
    daily_covid_deaths_plotly('US Conflicts, Daily Death Rates', usdf, 'linear', us_conflicts_df, 'Daily Deaths', 'Conflict', ' deaths/day')
    percentage_covid_deaths_plotly('US Conflicts as percentage of population', usdf, 'log', us_conflicts_df, '% US Population', 'Conflict', '% of population')
    total_covid_deaths_plotly('US Conflicts', usdf, 'log', us_conflicts_df, 'US Deaths', 'Conflict', ' deaths')
    daily_covid_deaths_plotly('US Leading Causes of Death, Daily Death Rates', usdf, 'linear', us_leading_causes_df, 'Daily Deaths', 'Leading Causes', ' deaths/day')
    percentage_covid_deaths_plotly('US Leading Causes of Death as percentage of population', usdf, 'log', us_leading_causes_df, '% US Population', 'Leading Causes', '% of population')
    total_covid_deaths_plotly('US Leading Causes of Death', usdf, 'log', us_leading_causes_df, 'US Deaths', 'Leading Causes', ' deaths')
    daily_covid_deaths_plotly('US Epidemics, Daily Death Rates', usdf, 'linear', us_epidemics_df, 'Daily Deaths', 'Epidemic', ' deaths/day')
    percentage_covid_deaths_plotly('US Epidemics as percentage of population', usdf, 'log', us_epidemics_df, '% US Population', 'Epidemic', '% of population')
    total_covid_deaths_plotly('US Epidemics', usdf, 'log', us_epidemics_df, 'US Deaths', 'Epidemic', ' deaths')
    daily_covid_deaths_plotly('US Disasters', usdf, 'linear', us_disasters_df, 'US Deaths', 'US Disasters', ' deaths')
    percentage_covid_deaths_plotly('US Disasters as percentage of population', usdf, 'log', us_disasters_df, '% US Population', 'US Disasters', '% of population')
    total_covid_deaths_plotly('US Disasters', usdf, 'log', us_disasters_df, 'US Deaths', 'US Disasters', ' deaths')

main()

