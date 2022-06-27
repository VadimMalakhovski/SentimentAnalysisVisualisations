"""
NAME
    app

DESCRIPTION
    # Program Name: app.py
    # Purpose: main app that executes most other scripts - like preprocess and execution of MLs and chart generation
    # Example Of: Functions of generating ML predictions and charts visualisations
"""

from flask import Flask, render_template
import pandas as pd
from collections import Counter
from traditional_ml import nb_score, lr_score, svc_score
from dnn import dnn_score
from vader import vader_score

app = Flask(__name__)


# Comparison Results Output of different Sentiment Analysis models
print('Comparison of SA models Naive Bayes, Logistic Regression, SVM and DNN with VADER')
# Results Table
results_table = [['Model Name', 'Accuracy'],
              ['Naive Bayes', nb_score],
              ['Logistics Regression   ', lr_score],
              ['Support Vector Machine', svc_score],
              ['Deep Neural Network', dnn_score],
              ['_______________________', '_____'],
              ['Vader', vader_score]]

column_width = max(len(entry) for row in results_table for entry in row) + 2
for row in results_table:
    print("".join(entry.ljust(column_width) for entry in row))


# load csv file with pandas
df = pd.read_csv("tweets_data/tweets_annotated_compiled.csv")

# Continents
# add total sentiment tweets per continent in this order 'EUR', 'NAM', 'AUS', 'ASI', 'SAM'
# sentiment lists with continents
negative_tweets_per_continent_list, neutral_tweets_per_continent_list, positive_tweets_per_continent_list = [], [], []
average_tweets_per_continent_list = []
full_continent_names_list = ['Europe', 'North America', 'Australia', 'Asia', 'South America']
continent_codes_list = ['EUR', 'NAM', 'AUS', 'ASI', 'SAM']

# sort sentiment tweets per continent and calculate averages
for continent in continent_codes_list:
    neg = len(df.loc[(df["Continent_Code"] == continent) & (df["Sentiment_Score"] == 0)])
    negative_tweets_per_continent_list.append(neg)
    neut = len(df.loc[(df["Continent_Code"] == continent) & (df["Sentiment_Score"] == 2)])
    neutral_tweets_per_continent_list.append(neut)
    pos = len(df.loc[(df["Continent_Code"] == continent) & (df["Sentiment_Score"] == 4)])
    positive_tweets_per_continent_list.append(pos)
    average_tweets_per_continent_list.append((neg + neut + pos) / 3)

# count total neg neu pos tweets
negative_number_of_tweets = len(df.loc[(df["Sentiment_Score"] == 0)])
neutral_number_of_tweets = len(df.loc[(df["Sentiment_Score"] == 2)])
positive_number_of_tweets = len(df.loc[(df["Sentiment_Score"] == 4)])

# get negative tweets' relative times posted
time_for_negative_tweets_df = df.loc[df['Sentiment_Score'] == 0, 'Relative_Time']
time_for_negative_tweets_list = time_for_negative_tweets_df.tolist()

# get neutral tweets' relative times posted
time_for_neutral_tweets_df = df.loc[df['Sentiment_Score'] == 2, 'Relative_Time']
time_for_neutral_tweets_list = time_for_neutral_tweets_df.tolist()

# get positive tweets' relative times posted
time_for_positive_tweets_df = df.loc[df['Sentiment_Score'] == 4, 'Relative_Time']
time_for_positive_tweets_list = time_for_positive_tweets_df.tolist()

# hours list range from 0-24
hours_list = [*range(1, 25, 1)]


# format data to a correct format accepted by scatter plot
def format_data_scatter(sentiment_value):
    if sentiment_value == "Negative":
        sentiment_value = 0
    elif sentiment_value == "Positive":
        sentiment_value = 4
    else:
        sentiment_value = 2

    # get the reach/retweets of sentiment
    continent_reach = df.loc[df['Sentiment_Score'] == sentiment_value, 'Reach'].tolist()
    continent_time_of_tweet = df.loc[df['Sentiment_Score'] == sentiment_value, 'Relative_Time'].tolist()
    # get the tweets reach with their relative time of day they were posted
    continent_reach_vs_time_of_tweet = [x for y in zip(continent_reach, continent_time_of_tweet) for x in y]

    # Split a List into Chunks of 2 as this is the format required for this particular highcharts graphs
    continent_reach_vs_time_of_tweet_formatted = list()
    chunk_size = 2
    for i in range(0, len(continent_reach_vs_time_of_tweet), chunk_size):
        continent_reach_vs_time_of_tweet_formatted.append(continent_reach_vs_time_of_tweet[i:i + chunk_size])
    return continent_reach_vs_time_of_tweet_formatted


# Home Page with Summary
@app.route('/')
def summary_graph(chartID='chart_ID', chart_type='line', chart_height=500):
    pageType = 'summary_graph'
    chart = {"renderTo": chartID, "type": chart_type, "height": chart_height}
    title = {"text": 'Summary Chart'}
    xAxis = {"categories": full_continent_names_list}
    yAxis = [{"title": {"text": 'Tweets'}, "min": "0"}]
    labels = {"items": [{"html": 'Sentiment Ratio', "style": {"left": '100px', "top": '2px',
                                                              "color": '(Highcharts.defaultOptions.title.style & &Highcharts.defaultOptions.title.style.color) | |' 'black'}}]}
    series = [{"type": 'column', "name": 'Negative', "data": negative_tweets_per_continent_list, "color": '#f45b5b'},
              {"type": 'column', "name": 'Neutral', "data": neutral_tweets_per_continent_list, "color": '#e4d354'},
              {"type": 'column', "name": 'Positive', "data": positive_tweets_per_continent_list, "color": '#90ed7d'},
              {"type": 'spline', "name": 'Average', "data": average_tweets_per_continent_list, "color": '#f7a35c',
               "marker": {"lineWidth": "2", "lineColor": "Highcharts.getOptions().colors[3]", "fillColor": 'white'}},
              {"type": 'pie', "name": 'Total Tweets per Continent',
               "data": [{"name": 'Negative', "y": negative_number_of_tweets, "color": '#f45b5b'},
                        {"name": 'Neutral', "y": neutral_number_of_tweets, "color": '#e4d354'},
                        {"name": 'Positive', "y": positive_number_of_tweets, "color": '#90ed7d'}], "size": "100",
               "center": [115, 70], "showInLegend": "false", "dataLabels": {"enabled": "false"}}]
    return render_template('index.html', chartID=chartID, chart=chart, labels=labels, series=series, title=title,
                           xAxis=xAxis, yAxis=yAxis, pageType=pageType)


# creating and formatting data for the heat map - has to be a 2D array in this format for each 24 hours
# this array will contain a single building block of 2D array - [hour, sentiment_type, sentiment_score] for each hour
single_array_unit = []
d2_array_compiled = []

# iterate x times as x hours
for hour in range(1, 25):
    sentiment = [0, 2, 4]
    sentiment_level = 0
    # iterate sentiment times
    for s in sentiment:
        sentiment_score = len(df.loc[(df["Sentiment_Score"] == s) & (df["Relative_Time"] == hour)])
        single_array_unit.append(hour - 1)
        single_array_unit.append(sentiment_level)
        sentiment_level += 1
        single_array_unit.append(sentiment_score)
        d2_array_compiled.append(single_array_unit)
        single_array_unit = []

# reformat time to 12 hours am and pm for better visualisation
def format_hours_am_pa(am_pm_period):
    return [str(n) + " " + am_pm_period for n in range(1, 13)]
am_hours_list = format_hours_am_pa("AM")
pm_hours_list = format_hours_am_pa("PM")
formatted_time_list = am_hours_list + pm_hours_list


@app.route('/temporal_graph')
def temporal_graph(chart2ID='chart2_ID', chart_type2='heatmap', chart_height2=500):
    pageType = 'temporal_graph'
    chart2 = {"renderTo": chart2ID, "type": chart_type2, "height": chart_height2}
    title2 = {"text": 'Spatial Chart'}
    # add 'am' + 'pm' - try 2 lists then concatenate
    xAxis2 = {"categories": formatted_time_list}
    yAxis2 = [{"title": "null", "categories": ['Negative', 'Neutral', 'Positive']}]
    colorAxis2 = {"min": "0", "minColor": '#FFFFFF', "maxColor": "#E30000"}  # "Highcharts.getOptions().colors[0]"}
    labels2 = {"items": [{"html": '', "style": {"left": '100px', "top": '2px',
                                                "color": '(Highcharts.defaultOptions.title.style & &Highcharts.defaultOptions.title.style.color) | |' 'black'}}]}
    series2 = [{"visible": "true", "name": 'Mood hour Data', "data": d2_array_compiled,"dataLabels": {"enabled": "true", "color": '#000000'}}]
    return render_template('index.html', chart2ID=chart2ID, chart2=chart2, labels2=labels2, series2=series2,
                           title2=title2, xAxis2=xAxis2, yAxis2=yAxis2, pageType=pageType, colorAxis2=colorAxis2)


# returns a list of retweets per specified continent
def calculate_reach(continent_code):
    # sample output 1d array of top n retweeted comments per continent = [363, 364, 364, 365, 365, 367, 375, 377, 378]
    reach_df = df.loc[df['Continent_Code'] == continent_code, 'Reach']
    reach_list = reach_df.tolist()
    reach_list.sort()
    # get top n influence's so that chart will be clearer and load faster
    top_n_reach_list = reach_list[-10:]
    return top_n_reach_list


# returns a list of dictionaries depending on the provided input so that format will be acceptable by highcharts
def generate_list_of_dict(enumerating_list, continent_code, world_data_list, key_name='value'):
    # reusable method - print depends on the input parameters - it will be a list of dictionaries
    for idx, val in enumerate(enumerating_list):
        if key_name == 'data':
            val = get_reach_each_individual(continent_codes_list[idx])
            continent_code = full_continent_names_list[idx]
        # creating list of dictionaries
        pairs = [('name', continent_code), (key_name, val)]
        dict(pairs)
        dict_pairs = dict([(k, v) for k, v in pairs])
        world_data_list.append(dict_pairs)
    return world_data_list


# returns/calls a method to generate a list of dictionaries with values of retweets for a specific continent
def get_reach_each_individual(continent_code):
    # sample output = [{'value': 425, 'name': 'EUR'}, {'value': 426, 'name': 'EUR'}]
    world_data_list = []
    return generate_list_of_dict(calculate_reach(continent_code), continent_code, world_data_list, 'value')


# returns/calls a method to generate a list of dictionaries acceptable for input into the highcharts
def generate_packedbubble_data():
    # sample output = [{"name": 'Europe',"data":[{'value': 425, 'name': 'EUR'}]}, {"name": 'North America', "data": [{'value': 425, 'name': 'NAM'}]]
    world_data_list = []
    return generate_list_of_dict(full_continent_names_list, continent_codes_list, world_data_list, 'data')


@app.route('/spatial_graph')
def spatial_graph(chart3ID='chart3_ID', chart_type3='packedbubble', chart_height3=500):
    pageType = 'spatial_graph'
    chart3 = {"renderTo": chart3ID, "type": chart_type3, "height": chart_height3}
    title3 = {"text": 'Spatial Chart'}
    xAxis3 = {"categories": "full_continent_names_list"}
    yAxis3 = [{"title": {"text": 'Tweets'}}]
    labels3 = {"items": [{"html": '', "style": {"left": '100px', "top": '2px',
                                                "color": '(Highcharts.defaultOptions.title.style & &Highcharts.defaultOptions.title.style.color) | |' 'black'}}]}
    plotOptions3 = {"dataLabels": {"style": {"color": 'red', "textOutline": 'none', "fontWeight": 'normal'}}}  # "packedbubble": {"minSize": '30%',"maxSize": '520%'},
    series3 = generate_packedbubble_data()
    return render_template('index.html', chart3ID=chart3ID, chart3=chart3, plotOptions3=plotOptions3, labels3=labels3,
                                         series3=series3, title3=title3, xAxis3=xAxis3, yAxis3=yAxis3, pageType=pageType)

# word cloud top 100 words + few this dataset related
top_100_words_list = \
    ["the", "of", "to", "and", "a", "in", "is", "it", "you", "that", "he", "was", "for", "on", "are", "with", "as", "i",
     "his", "they", "be", "at", "one", "have", "this", "from", "or", "had", "by", "not", "but", "what", "some", "we",
     "can", "out", "other", "were", "all", "there", "when", "up", "use", "your", "how", "said", "an", "each", "she",
     "which", "do", "their", "if", "will", "about", "many", "then", "them", "write", "would", "like", "so", "these",
     "her", "long", "make", "thing", "see", "him", "two", "has", "look", "more", "day", "could", "go", "come", "did",
     "number", "sound", "no", "most", "my", "over", "know", "water", "than", "call", "first", "who", "may", "down",
     "side", "been", "now", "find", "still", "going", "get", "via", "elon", "musk"]

# get most frequent words used
most_frequent_words = Counter(" ".join(df["Tweets"]).split()).most_common(100)
top_words_list = []
top_words_frequency_list = []

for item in most_frequent_words:
    if item[0] not in top_100_words_list:
        top_words_list.append(item[0])
        top_words_frequency_list.append(item[1])

# gather the list of dictionaries for all the available words with their frequency
# format sample output: [{"name": "Lorem","weight": "1"}, {"name": "ipsum","weight": "1"}, {"name": "blah","weight": "4"}]
word_cloud_data_list = []
for idx, val in enumerate(top_words_list):
    pairs = [('name', val), ('weight', top_words_frequency_list[idx])]
    dict(pairs)
    d = dict([(k, v) for k, v in pairs])
    word_cloud_data_list.append(d)

@app.route('/word_cloud')
def word_cloud(chartID='chart_ID', chart_type='wordcloud', chart_height=500):
    pageType = 'word_cloud'
    chart = {"renderTo": chartID, "type": chart_type, "height": chart_height}
    title = {"text": 'Wordcloud of Tweets'}
    xAxis = {"categories": full_continent_names_list}
    yAxis = [{"title": {"text": 'Tweets'}, "min": "0"}]
    labels = {"items": [{"html": '', "style": {"left": '100px', "top": '2px',
                                               "color": '(Highcharts.defaultOptions.title.style & &Highcharts.defaultOptions.title.style.color) | |' 'black'}}]}
    series = [{"type": 'wordcloud', "name": 'Occurrences', "data": word_cloud_data_list}]
    return render_template('index.html', chartID=chartID, chart=chart, labels=labels, series=series, title=title,
                                          xAxis=xAxis, yAxis=yAxis, pageType=pageType)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080, passthrough_errors=True)
