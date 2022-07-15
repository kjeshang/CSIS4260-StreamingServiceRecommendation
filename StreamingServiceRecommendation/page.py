# Setup -------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc

# Import the data ----------------------------------------------------
df = pd.read_excel("streaming_service_titles.xlsx", index_col=False);

# Prepare the data ---------------------------------------------------
techniqueSelector = [
    {
        "label":"TFIDF",
        "value":0
    },
    {
        "label":"Unigram",
        "value":1
    },
    {
        "label":"Bigram",
        "value":2
    },
    {
        "label":"Trigram",
        "value":3
    }
];

# Create elements of the webpage -------------------------------------
heading = [html.H1("Streaming Service Recommendation System", style={"font-weight":"bold"})];

titleDropdownSelector = [
    html.P("Title", style={"font-weight":"bold"}),
    dcc.Dropdown(
        id="title_select",
        options=[{"label":x, "value":x} for x in df["title"]],
        searchable=True, 
        placeholder="Select a title"
    )
];

streamingServiceChecklistFilter = [
    html.P("Streaming Service", style={"font-weight":"bold"}),
    dcc.Checklist(
        id="streaming_service_select",
        options=[{"label":x, "value":x} for x in df["Streaming Service"].unique().tolist()],
        value=df["Streaming Service"].unique().tolist(),
        inputStyle={"margin-right":"5px", "margin-left":"15px"}
    )
];

techniqueRadioButtonSelector = [
    html.P("Text Mining Technique", style={"font-weight":"bold"}),
    dcc.RadioItems(
        id="technique_select",
        options=[{"label":x.get("label"), "value":x.get("value")} for x in techniqueSelector],
        value=techniqueSelector[0].get("value"),
        inputStyle={"margin-right":"5px", "margin-left":"15px"}
    )
];

selectedTitleInformation = [html.Div(id="title_out")];

recommendationResultsInTable = [
    html.P("Recommendation Results", style={"font-weight":"bold"}),
    html.Div(id="recommendation_results")
];

streamingServiceRecommendationChart = [html.Div(id="streaming_service_chart")];

recommendedTitleInformation = [html.Div(id="tbl_out")];

# Page Layout --------------------------------------------------------
pageStructure = [
    dbc.Row(children=heading),
    html.Br(),
    dbc.Row(children=[
        dbc.Col(children=[
            dbc.Card(children=[
                dbc.CardBody(children=titleDropdownSelector)
            ])
        ], width=5),
        dbc.Col(children=[
            dbc.Card(children=[
                dbc.CardBody(children=streamingServiceChecklistFilter)
            ])
        ], width=4),
        dbc.Col(children=[
            dbc.Card(children=[
                dbc.CardBody(children=techniqueRadioButtonSelector)
            ])
        ], width=3)
    ]),
    html.Br(),
    dbc.Row(children=[
        dbc.Col(children=selectedTitleInformation),
        dbc.Col(children=recommendedTitleInformation)
    ]),
    html.Br(),
    dbc.Row(children=recommendationResultsInTable),
    html.Br(),
    dbc.Row(children=streamingServiceRecommendationChart)
];

pageLayout = dbc.Container(children=pageStructure, fluid=True);

# Callbacks ----------------------------------------------------------
def get_dataframeNLP(title_select, streaming_service_select):
    dff_temp = df.copy();
    dff_titleSelect = dff_temp[dff_temp["title"] == title_select];
    dff_streamingServiceSelect = dff_temp[dff_temp["Streaming Service"].isin(streaming_service_select)];
    dff = pd.concat([dff_titleSelect, dff_streamingServiceSelect]);
    dff = dff.drop_duplicates();
    dff = dff.sort_values(by=["title","release_year","rating","genres","Streaming Service"]);
    dff = dff.reset_index();
    dff = dff[dff.columns.tolist()[1:]]; 
    return dff;

def get_recommendations(dataframe, title_select, indices, cosine_sim):
    idx = indices[title_select];
    sim_scores = list(enumerate(cosine_sim[idx]));
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True);
    sim_scores = sim_scores[1:11];
    movie_indices = [i[0] for i in sim_scores];
    dataframe1 = dataframe[["title","description","Streaming Service"]].iloc[movie_indices];
    similarityScores = [];
    for i in range(len(sim_scores)):
        similarityScores.append(round(sim_scores[i][1], 4));
    dataframe1["Similarity Score"] =  similarityScores;
    dataframe1 = dataframe1.reset_index().rename(columns={"index":"id"});
    return dataframe1;

def get_streamingServiceRecommendationChart(dataframeRecommendation):
    data = [];
    for service in dataframeRecommendation["Streaming Service"].unique().tolist():
        row = [];
        row.append(service);
        row.append(len(dataframeRecommendation[dataframeRecommendation["Streaming Service"] == service]));
        data.append(row);
    dataframeStreamingService = pd.DataFrame(data, columns=["Streaming Service","Count"]);
    fig = px.bar(dataframeStreamingService, x="Streaming Service", y="Count", color="Streaming Service");
    return fig;

@callback(
    Output("title_out", "children"),
    Input("title_select", "value")
)
def get_selectedTitleInformation(title_select):
    selectedTitleInfo = html.P("No title selected");
    if title_select != None:
        idx = df.index[df["title"] == title_select].tolist()[0];
        selectedTitleInfo = dbc.Card([
            dbc.CardHeader("Selected Title"),
            dbc.CardBody([
                html.H4(df.loc[idx, "title"] + " (" + str(df.loc[idx, "type"]) + ", Released: " + str(df.loc[idx, "release_year"]) + ")"),
                html.P(str(df.loc[idx, "description"])),
                html.P("Rating: " + str(df.loc[idx, "rating"])),
                html.P("Genres: " + str(df.loc[idx, "genres"]))
            ]),
            dbc.CardFooter(df.loc[idx, "Streaming Service"])
        ]);
    return selectedTitleInfo;

@callback(
    Output("recommendation_results", "children"),
    Output("streaming_service_chart", "children"),
    [
        Input("title_select", "value"),
        Input("streaming_service_select", "value"),
        Input("technique_select", "value")
    ],
)
def get_recommendationResults(title_select, streaming_service_select, technique_select):
    dashTable = html.P("No recommendation results");
    chart = html.P("");
    if (title_select != None) & (len(streaming_service_select) != 0):
        print(title_select);
        print(streaming_service_select);
        dff = get_dataframeNLP(title_select, streaming_service_select);
        
        # tfidf = TfidfVectorizer(strip_accents="ascii", stop_words="english", min_df=0.0005, sublinear_tf=True);
        # tfidf = TfidfVectorizer(max_df=.65, min_df=1, stop_words="english", use_idf=True, norm=None);
        # tfidf = TfidfVectorizer(stop_words="english", min_df=0.005, sublinear_tf=True);
        # tfidf_matrix = tfidf.fit_transform(dff["Textual Info"]);
        # cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix);
        if technique_select == 0:
            vectorizer = TfidfVectorizer(stop_words="english", min_df=0.005, sublinear_tf=True);
            matrix = vectorizer.fit_transform(dff["Textual Info"]);
            cosine_sim = linear_kernel(matrix, matrix);
        else:
            for num in range(1,4):
                if num == technique_select:
                    n = num;
                    print(n);
                    vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\b[a-zA-Z]{3,}\b', ngram_range=(n, n));
            matrix = vectorizer.fit_transform(dff["Textual Info"]);
            cosine_sim = cosine_similarity(matrix, matrix);
        # matrix = vectorizer.fit_transform(dff["Textual Info"]);
        # cosine_sim = linear_kernel(matrix, matrix);
        indices = pd.Series(dff.index, index=dff["title"]).drop_duplicates();

        # Recommendation Results:
        df_recommendations = get_recommendations(dff, title_select, indices, cosine_sim=cosine_sim);
        dashTable = dash_table.DataTable(
            style_cell={
                'whiteSpace':'normal',
                'height':'auto'
            },
            data=df_recommendations.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_recommendations.columns if i != "id"],
            style_cell_conditional=[
                {
                    'if': {'column_id': c},
                    'textAlign': 'left'
                } for c in df_recommendations.columns.tolist()[:-1]
            ],
            id="tbl"
        );

        # Streaming Service Chart:
        fig = get_streamingServiceRecommendationChart(df_recommendations);
        chart = dcc.Graph(figure=fig, config={"displayModeBar":False});
            
    return dashTable, chart;

@callback(
    Output("tbl_out", "children"),
    [
        Input("tbl", "active_cell"),
        Input("title_select", "value"),
        Input("streaming_service_select", "value")
    ],
)
def get_recommendedTitleInformation(active_cell, title_select, streaming_service_select):
    recommendedTitleInfo = html.P("No recommended title selected");
    print(active_cell);
    if (bool(active_cell) == True) & (title_select != None) & (len(streaming_service_select) != 0):
        # print(str(active_cell));
        row_id = active_cell.get("row_id");
        dff = get_dataframeNLP(title_select, streaming_service_select);
        print(dff.loc[row_id, "title"]);
        print(dff.loc[row_id, "type"]);
        print(dff.loc[row_id, "release_year"]);
        recommendedTitleInfo = dbc.Card([
            dbc.CardHeader("Recommended Title"),
            dbc.CardBody([
                html.H4(dff.loc[row_id, "title"] + " (" + str(dff.loc[row_id, "type"]) + ", Released: " + str(dff.loc[row_id, "release_year"]) + ")"),
                html.P(str(dff.loc[row_id, "description"])),
                html.P("Rating: " + str(dff.loc[row_id, "rating"])),
                html.P("Genres: " + str(dff.loc[row_id, "genres"]))
            ]),
            dbc.CardFooter(dff.loc[row_id, "Streaming Service"])            
        ]);
    return recommendedTitleInfo;