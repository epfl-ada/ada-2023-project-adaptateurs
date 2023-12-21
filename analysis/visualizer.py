import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

mpl.rcParams["figure.dpi"] = 300
color_M = "darkblue"  # ~rgb(0,0,139)
color_F = "crimson"  # ~rgb(182,28,66)
color_G = "black"
color_B = "darkorange"


def visualize_year_distribution(movies, style="darkgrid"):
    """
    Visualize the distribution of movies across years.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing movie data.
    style (str): Style of the plot. Default is "darkgrid".

    Returns:
    None
    """

    plt.figure(figsize=(8, 4))

    unique_movies = movies.drop_duplicates(subset="movie_title")
    mean = unique_movies.groupby("year")["wikiID"].count().mean()

    sns.set_style(style)
    sns.histplot(unique_movies.movie_release_date, bins=100, color=color_M)

    plt.title("Distribution of Movies Across Years")
    plt.xlabel("Release Year")
    plt.ylabel("Number of Movies")
    plt.show()

    print("Total number of movies: ", len(unique_movies))
    print(f"Mean number of movies per year: {round(mean)}")


def visualize_year_distribution_HTML(
    movies, output_html="html_plots/movie_distribution.html"
):
    """
    Visualize the distribution of movies across years using Plotly.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing movie data.
    output_html (str): The name of the output HTML file.

    Returns:
    None
    """

    # Preparing the data
    unique_movies = movies.drop_duplicates(subset="movie_title")
    mean = unique_movies.groupby("year")["wikiID"].count().mean()

    # Create a histogram using Plotly
    fig = px.histogram(
        unique_movies,
        x="movie_release_date",
        nbins=100,
        color_discrete_sequence=[color_G],
        opacity=0.5,
    )

    # Update layout for the plot
    fig.update_layout(
        title_text="Number of Movies released per year",
        xaxis_title="Release Year",
        yaxis_title="Number of Movies",
    )

    # Display the plot
    fig.show()

    # Export the plot to an HTML file
    fig.write_html(output_html)

    # Print additional information
    print("Total number of movies: ", len(unique_movies))
    print(f"Mean number of movies per year: {round(mean)}")


def visualize_missing_values(movies, style="darkgrid"):
    """
    Visualize the percentage of missing values in each column of a pandas DataFrame.

    Parameters:
    movies (pandas.DataFrame): The DataFrame to analyze.
    style (str): The style of the plot. Default is "darkgrid".

    Returns:
    None
    """

    missing_values = movies.isnull().sum()
    missing_percentage = (movies.isnull().sum() / len(movies)) * 100
    missing_values_df = pd.DataFrame(
        {"Missing Values": missing_values, "Percentage": missing_percentage}
    ).sort_values(by="Percentage", ascending=False)
    missing_values_df = missing_values_df[missing_values_df.Percentage > 0]

    sns.set_style(style)
    missing_values_df.Percentage.plot(
        kind="barh", figsize=(8, 6), color=color_M, legend=False, alpha=0.5
    )

    plt.title("Percentage of Missing Values in Each Column")
    plt.ylabel("Columns")
    plt.xlabel("Percentage Missing")
    plt.show()


def visualize_gender_distribution(movies):
    roles = ["actor_gender", "director_gender", "producer_gender"]
    
    # Define color map for genders
    color_map = {'M': color_M, 'F': color_F}  # Ensure these are defined, e.g., color_M = 'blue', color_F = 'red'

    # Create figure for the stacked bar chart
    fig, ax = plt.subplots(figsize=(6, 3))

    # Variables to hold the bottom values for the stacked bars
    bar_bottoms = np.zeros(len(roles))

    # Loop through each gender and plot
    for gender in ['F', 'M']:  # Ensuring consistent order
        gender_percentages = []
        for role in roles:
            gender_count = movies[role].value_counts().get(gender, 0)
            total_count = movies[role].count()
            percentage = (gender_count / total_count * 100) if total_count > 0 else 0
            gender_percentages.append(percentage)

        ax.bar(
            roles,
            gender_percentages,
            bottom=bar_bottoms,
            color=color_map.get(gender, 'gray'),
            label=f"{gender}"
        )
        bar_bottoms += np.array(gender_percentages)

    ax.set_ylabel("Percentage (%)")
    ax.legend()

    plt.suptitle("Gender Distribution in Main Roles (Percentage)")
    plt.show()


def visualize_gender_distribution_HTML(movies, output_html='html_plots/gender_distribution_bar.html'):
    roles = ["actor_gender", "director_gender", "producer_gender"]
    
    # Create a subplot figure with 1 row and len(roles) columns
    fig = make_subplots(rows=1, cols=len(roles))
    
    # Define color map for genders
    color_map = {'M': color_M, 'F': color_F}

    # Add traces for each gender
    for i, role in enumerate(roles, start=1):
        gender_counts = movies[role].value_counts().reset_index()
        gender_counts.columns = [role, 'count']
        total_count = gender_counts['count'].sum()
        gender_counts['percentage'] = (gender_counts['count'] / total_count * 100).round(2)

        for gender in sorted(gender_counts[role], reverse=False):  # Sorts 'F' before 'M'
            show_legend = i == 1  # Show legend only for the first subplot
            fig.add_trace(go.Bar(
                x=[role.replace("_gender", "").title()],
                y=[gender_counts.loc[gender_counts[role] == gender, 'count'].values[0]],
                text=f"{gender_counts.loc[gender_counts[role] == gender, 'percentage'].values[0]}%",
                textposition='inside',
                marker_color=color_map[gender] if gender in color_map else 'gray',
                name=gender,
                showlegend=show_legend,
                opacity=0.8
            ), 1, i)

    # Update layout
    fig.update_layout(
        title_text="Gender distribution in the main roles",
        barmode='stack',
    )
    # Update x-axis and y-axis labels
    for i in range(1, len(roles) + 1):
        fig.update_xaxes(tickfont=dict(size=14), row=1, col=i)
    fig.update_yaxes(title_text='Count')

    # Display the plot
    fig.show()

    # Export the plot to an HTML file
    fig.write_html(output_html)


def visualize_actors_gender_proportion(movies):
    """
    Visualize the distribution of the number of actors in movies as a stacked bar chart using Matplotlib.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing the movies data.

    Returns:
    None
    """
    # Count the number of actors by year and gender
    actor_counts = (
        movies.groupby(["year", "actor_gender"]).count()["actor_name"].reset_index()
    )

    # Calculate the total actors per year
    total_actors_per_year = movies.groupby("year")["actor_name"].count()

    # Calculate the percentage
    actor_counts["percentage"] = actor_counts.apply(
        lambda row: (row["actor_name"] / total_actors_per_year[row["year"]]) * 100,
        axis=1,
    )

    # Pivot for stacked bar plot
    pivot_df = actor_counts.pivot(index='year', columns='actor_gender', values='percentage')

    # Plotting
    plt.figure(figsize=(6, 3))
    pivot_df.plot(kind='bar', stacked=True, color=[color_F, color_M], ax=plt.gca())

    plt.title("Mean Proportion of Actors and Actresses in Movies by Year")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Actors (%)")

    # Show 1 year every 5 years
    plt.xticks(np.arange(0, len(pivot_df.index), 5), pivot_df.index[::5], rotation=90)
    plt.ylim(0, 100)  # Set y-axis to range from 0 to 100%
    plt.legend(title="Actor Gender")
    plt.grid(True, which='major', linestyle='-', linewidth='0.5', color=color_G)


def visualize_actors_gender_proportion_HTML(
    movies, year_range=[], output_html="html_plots/actors_gender_proportion.html"
):
    """
    Visualize the distribution of the number of actors in movies as a piled histogram

    Parameters:
    movies (pandas.DataFrame): DataFrame containing the movies data.
    output_html (str): The name of the output HTML file.
    """
    # Count the number of actors by year and gender
    actor_counts = (
        movies.groupby(["year", "actor_gender"]).count()["actor_name"].reset_index()
    )

    # Calculate the total actors per year
    total_actors_per_year = movies.groupby("year")["actor_name"].count()

    # Calculate the percentage
    actor_counts["percentage"] = actor_counts.apply(
        lambda row: (row["actor_name"] / total_actors_per_year[row["year"]]),
        axis=1,
    )

    # Set color for each gender
    color_discrete_map = {"F": color_F, "M": color_M}

    # Create a piled histogram using Plotly
    fig = px.bar(
        actor_counts,
        x="year",
        y="percentage",
        color="actor_gender",
        title="Mean proportion of Actors and Actresses in movies by year",
        labels={
            "percentage": "Percentage of Actors (%)",
            "actor_gender": "Actor Gender",
            "year": "Year",
        },
        color_discrete_map=color_discrete_map,
        opacity=0.8,
    )

    fig.update_layout(bargap=0, yaxis_tickformat='0%')
    fig.update_xaxes(range=year_range)
    # Set y-axis to range from 0 to 100%
    fig.update_yaxes(range=[0, 1], showgrid=True, gridwidth=1, gridcolor=color_G)

    # Display the plot
    fig.show()

    # Export the plot to an HTML file
    fig.write_html(output_html)


# Not used in P3
def visualize_gender_prop(movies, style="darkgrid"):
    movies = movies.loc[movies["actor_gender"].isin(["F", "M"])].copy(deep=True)
    male_female_counts = (
        movies.groupby(["wikiID", "actor_gender"])["character_name"]
        .nunique()
        .unstack(fill_value=0)
    )
    male_female_counts = male_female_counts.join(
        movies.groupby("wikiID")["character_name"].nunique().rename("total_char")
    ).assign(
        percents_of_female=lambda x: x["F"] / x["total_char"] * 100,
        percents_of_male=lambda x: x["M"] / x["total_char"] * 100,
    )

    male_female_counts.rename(
        columns={"F": "female_char", "M": "male_char"}, inplace=True
    )
    male_female_counts = male_female_counts.merge(
        movies[["year", "wikiID"]].drop_duplicates(), on="wikiID", how="left"
    )

    male_female_counts.dropna(
        inplace=True, subset=["percents_of_female", "percents_of_male"]
    )

    _df = pd.DataFrame(
        {
            "Female actresses": male_female_counts["percents_of_female"],
            "Male actors": male_female_counts["percents_of_male"],
        }
    )

    # Plot results into two distinct plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8), sharey=True)

    sns.set_style(style)

    sns.histplot(
        data=_df,
        ax=ax,
        stat="count",
        multiple="stack",
        bins=20,
        alpha=1,
        palette=[color_F, color_M],
        element="bars",
        legend=True,
    )

    fig.supxlabel("Percentage of characters per movie", fontsize=16)
    fig.suptitle(
        "Histograms of the percentage of Female and Male characters per movie",
        fontsize=20,
    )

    plt.tight_layout()
    plt.show()
    return male_female_counts


def visualize_regression(movies_gender_prop, style="darkgrid"):
    """
    Visualize the correlation between the percentage of women actors in movies and their average rating.

    Args:
        movies_gender_prop (pandas.DataFrame): A DataFrame containing the percentage of women actors and the average rating of movies.
        style (str, optional): The style of the plot. Defaults to "darkgrid".
    """

    plt.figure(figsize=(10, 5))

    sns.set_style(style)
    sns.regplot(
        x=movies_gender_prop["Female_Actors_Per_Film"]/movies_gender_prop["Total_Actors_Per_Film"],
        y=movies_gender_prop["vote_average"],
        color=color_G,
        scatter_kws={"s": 2, "alpha": 0.3},
        line_kws={"color": color_F},
    )

    plt.ylabel("Average Rating", fontsize=14)
    plt.xlabel("Percentage of women actresses", fontsize=14)
    plt.title("Correlation between Rating and proportion of Women", fontsize=20)

    plt.show()


def visualize_age_distribution_by_gender(movies, style="darkgrid"):
    """
    Visualize the distribution of actor ages by gender using a boxplot.

    Args:
    movies (pandas.DataFrame): A DataFrame containing movie data with columns 'actor_gender' and 'actor_age_at_movie_release'.

    Returns:
    None
    """

    plt.figure(figsize=(15, 6))
    sns.set_style(style)
    sns.boxplot(
        x="actor_gender",
        y="actor_age_at_movie_release",
        data=movies,
        palette=[color_F, color_M],
    )
    plt.title("Distribution of Actor Ages by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Age")
    plt.show()


def visualize_age_evolution(movies, style="darkgrid"):
    """
    Visualize the evolution of actors' age over the years.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".

    Returns:
    None
    """

    plt.figure(figsize=(10, 6))
    sns.set_style(style)
    sns.lineplot(
        data=movies,
        x="year",
        y="actor_age_at_movie_release",
        hue="actor_gender",
        errorbar=("ci", 95),
        palette=[color_F, color_M],
    )
    plt.title("Evolution of actors' age over the years")
    plt.xlabel("Year")
    plt.ylabel("Age")
    plt.show()


def visualize_age_evolution_HTML(
    movies, year_range=[], output_html="html_plots/age_evolution_plot.html"
):
    """
    Visualize the average evolution of actors' and actresses' age over the years using Plotly and export as HTML.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies, actors, and their ages.
    output_html (str): The name of the output HTML file.

    Returns:
    None
    """

    # Calculate the median age of actors and actresses for each year
    median_ages = (
        movies.groupby(["year", "actor_gender"])["actor_age_at_movie_release"]
        .median()
        .reset_index()
    )

    # Calculate the age difference for each year
    age_diff = median_ages.pivot(
        index="year", columns="actor_gender", values="actor_age_at_movie_release"
    )
    age_diff["age_difference"] = age_diff["M"] - age_diff["F"]

    color_discrete_map = {"F": color_F, "M": color_M}

    # Create the plot using Plotly
    fig = px.line(
        median_ages,
        x="year",
        y="actor_age_at_movie_release",
        color="actor_gender",
        title="Evolution of the median age of Actors and Actresses over the Years",
        labels={
            "actor_age_at_movie_release": "Median Age",
            "actor_gender": "Actor Gender",
            "year": "Year",
        },
        color_discrete_map=color_discrete_map,
    )

    # Add the age difference line
    fig.add_trace(
        go.Scatter(
            x=age_diff.index,
            y=age_diff["age_difference"],
            mode="lines",
            name="Age Difference (M - F)",
            line=dict(dash="dot", color=color_G),
        )
    )

    fig.update_xaxes(range=year_range)
    fig.update_yaxes(range=[0, 50])

    # Display the plot
    fig.show()

    # Export the plot to an HTML file
    fig.write_html(output_html)


def visualize_bechdel_distribution(movies_with_bechdel, style="darkgrid"):
    """
    Visualize the distribution of Bechdel test scores across movies.
    Args: movies_with_bechdel (pandas.DataFrame): A DataFrame containing movie data with a 'bechdel_rating' column.
    Returns: None
    """
    df_unique_movies = movies_with_bechdel.drop_duplicates(
        subset=["movie_title", "year"]
    )

    # Group by year and Bechdel grade, then count the number of movies
    grouped_bechdel = (
        df_unique_movies.groupby(["year", "bechdel_rating"])
        .size()
        .unstack(fill_value=0)
    )

    # Determine the complete range of years
    all_years = range(grouped_bechdel.index.min(), grouped_bechdel.index.max() + 1)

    # Create a DataFrame with all years and merge
    all_years_df = pd.DataFrame(index=all_years)

    grouped = all_years_df.merge(
        grouped_bechdel, left_index=True, right_index=True, how="left"
    ).fillna(0)

    # Define a red-to-green colormap
    cmap = plt.get_cmap("RdYlGn")
    colors = cmap(np.linspace(0, 1, len(grouped.columns)))

    # Create a stacked bar chart
    grouped.plot(kind="bar", stacked=True, figsize=(15, 7), color=colors)

    # Filter x-axis ticks to display one year every five years
    tick_labels = [year if year % 5 == 0 else "" for year in all_years]
    plt.xticks(range(len(grouped)), tick_labels, rotation=0)

    # Adding labels and title
    plt.xlabel("Year")
    plt.ylabel("Number of Movies")
    plt.title("Number of Movies by Bechdel test score per Year")
    plt.legend(title="Bechdel score")

    # Show the plot
    plt.show()
    return df_unique_movies


def visualize_feminity_score_distribution(movies, style="darkgrid"):
    """
    Visualize the distribution of the femininity score across movies.

    Args:
    movies (pandas.DataFrame): A DataFrame containing movie data with a 'femininity_score' column.

    Returns:
    None
    """

    plt.figure(figsize=(8, 4))
    sns.set_style(style)
    sns.histplot(movies.feminity_score, bins=18, color=color_G, alpha=0.5, log=True)
    plt.title("Distribution of Femininity Score Across Movies")
    plt.xlim(0, 0.18)
    plt.xlabel("Femininity Score")
    plt.ylabel("Number of Movies")
    plt.show()


def visualize_masculinity_score_distribution(movies, style="darkgrid"):
    """
    Visualize the distribution of the femininity score across movies.

    Args:
    movies (pandas.DataFrame): A DataFrame containing movie data with a 'femininity_score' column.

    Returns:
    None
    """

    plt.figure(figsize=(8, 4))
    sns.set_style(style)
    sns.histplot(movies.masculinity_score, bins=30, color=color_G, alpha=0.5, log=True)
    plt.title("Distribution of Masculinity Score Across Movies")
    plt.xlim(0, 0.18)
    plt.xlabel("Masculinity Score")
    plt.ylabel("Number of Movies")
    plt.show()


def visualize_wordcloud_roles(actor_with_role):
    role_women = actor_with_role.loc[actor_with_role["actor_gender"] == "F"].copy(
        deep=True
    )
    role_men = actor_with_role.loc[actor_with_role["actor_gender"] == "M"].copy(
        deep=True
    )

    women_counts = role_women["role"].value_counts().head(5)
    men_counts = role_men["role"].value_counts().head(5)

    # Create WordClouds for women and men
    women_wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Oranges"
    ).generate_from_frequencies(women_counts)
    men_wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Oranges"
    ).generate_from_frequencies(men_counts)

    # Plotting the WordClouds
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(women_wordcloud, interpolation="bilinear")
    plt.title("Women Main Characters", color="black", fontsize=16)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(men_wordcloud, interpolation="bilinear")
    plt.title("Men Main Characters", color="black", fontsize=16)
    plt.axis("off")

    plt.show()
    return


def visualize_director_producer_actor_gender_correlation_HTML(movies):
    """
    Visualize the correlation between the presence of a female director and/or a female producer and the proportion of female actors in movies using box plots with Plotly, and export as HTML.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies, directors, producers, and actors.

    Returns:
    None
    """

    # Calculate the count of actors for each movie
    actor_counts = movies.groupby('wikiID').size()
    female_actor_counts = movies[movies['actor_gender'] == 'F'].groupby('wikiID').size()

    # Calculate the proportion of female actors
    female_actor_proportion = (female_actor_counts / actor_counts).rename('female_actor_proportion')

    # Create binary columns for the presence of a female director and a female producer
    movies['has_female_director'] = movies['director_gender'] == 'F'
    movies['has_female_producer'] = movies['producer_gender'] == 'F'
    
    # Merge the proportion of female actors into the movies DataFrame
    movies = movies.merge(female_actor_proportion, on='wikiID', how='left')
    
    # Drop duplicates since there can be multiple actors per movie
    movies.drop_duplicates(subset='wikiID', inplace=True)

    # Create a categorical variable for the four groups
    conditions = [
        (~movies['has_female_director'] & ~movies['has_female_producer']),
        (~movies['has_female_director'] & movies['has_female_producer']),
        (movies['has_female_director'] & ~movies['has_female_producer']),
        (movies['has_female_director'] & movies['has_female_producer'])
    ]
    choices = ['No Female Director/Producer', 'Female Producer Only', 'Female Director Only', 'Both Female Director and Producer']
    movies['category'] = np.select(conditions, choices)

    # Sort categories based on the median proportion of female actors
    category_medians = movies.groupby('category')['female_actor_proportion'].median().sort_values()
    sorted_categories = category_medians.index.tolist()

    # Create a color gradient from red to green
    color_gradient = {
        sorted_categories[0]: 'red',    # Lowest median
        sorted_categories[1]: 'orange',
        sorted_categories[2]: 'lightgreen',
        sorted_categories[3]: 'green'   # Highest median
    }

    # Create the plot with Plotly
    fig = px.box(movies, x='category', y='female_actor_proportion', title="Proportion of Female Actors by Presence of Female Directors and Producers",
                 labels={'female_actor_proportion': 'Proportion of Female Actors'},
                 category_orders={'category': sorted_categories},
                 color='category',
                 color_discrete_map=color_gradient,
                 notched=True)

    # Loop through the traces and update the fillcolor, median line color, and box border color
    for trace in fig.data:
        trace.update(fillcolor=trace.marker.color)
        
    # Set the median line color and box border color to black
    fig.update_traces(line=dict(color='black', width=1))

    # Remove the x-axis category labels
    fig.update_xaxes(showticklabels=False)

    # Update layout for better readability and set the figure size
    fig.update_layout(
        xaxis_tickangle=-45
    )
    
    # Export the plot to an HTML file
    fig.write_html("html_plots/director_producer_actor_gender_correlation_proportion.html")
    
    fig.show()




def visualize_producer_gender_proportion_HTML(movies, year_range=[], output_html='html_plots/producer_gender_proportion.html'):
    """
    Visualize the count of movies with male and female producers over a specified range of years using Plotly, and export as HTML.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and producers.
    start_year (int): The start year of the range to consider.
    end_year (int): The end year of the range to consider.
    year_range (int): The range of years to average over. Default is 5.
    output_html (str): The name of the output HTML file.

    Returns:
    None
    """
    start_year = year_range[0]
    end_year = year_range[1]
    # Filter the movies DataFrame to only include movies within the specified year range
    movies = movies[(movies["year"] >= start_year) & (movies["year"] <= end_year)]

    # Group by year range and producer gender
    
    gender_counts = (
        movies.groupby(["year", "producer_gender"]).size().unstack(fill_value=0)
    )

    gender_counts["Female proportion"] = gender_counts["F"] / (gender_counts["F"] + gender_counts["M"])
    gender_counts["Male proportion"] = gender_counts["M"] / (gender_counts["F"] + gender_counts["M"])

    # Create the figure with Plotly
    fig = go.Figure()

    # Add female producers on top of the male producers bar trace
    fig.add_trace(go.Bar(
        x=gender_counts.index,
        y=gender_counts["Female proportion"],
        name="Female",
        marker_color=color_F,
        opacity=0.8,
    ))
    # Add male producers as a bar trace
    fig.add_trace(go.Bar(
        x=gender_counts.index,
        y=gender_counts["Male proportion"],
        name="Male",
        marker_color=color_M,
        opacity=0.8
    ))

    # Update the layout to stack the bars
    fig.update_layout(
        barmode='stack',
        title=f"Proportion of Movies with male and female Producers from {start_year} to {end_year}",
        xaxis_title="Year",
        yaxis_title="Count of Movies",
        legend_title="Producer Gender",
        yaxis_tickformat='0%', 
        bargap=0
    )
    
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_G)

    # Export the plot to an HTML file
    fig.write_html(output_html)

    # Optionally, display the plot in the notebook if you're using Jupyter
    fig.show()


def visualize_director_gender_proportion_HTML(movies, year_range=[], output_html='html_plots/director_gender_proportion.html'):
    """
    Visualize the count of movies with male and female producers over a specified range of years using Plotly, and export as HTML.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and producers.
    start_year (int): The start year of the range to consider.
    end_year (int): The end year of the range to consider.
    year_range (int): The range of years to average over. Default is 5.
    output_html (str): The name of the output HTML file.

    Returns:
    None
    """
    start_year = year_range[0]
    end_year = year_range[1]
    # Filter the movies DataFrame to only include movies within the specified year range
    movies = movies[(movies["year"] >= start_year) & (movies["year"] <= end_year)]

    # Group by year range and producer gender
    
    gender_counts = (
        movies.groupby(["year", "director_gender"]).size().unstack(fill_value=0)
    )

    gender_counts["Female proportion"] = gender_counts["F"] / (gender_counts["F"] + gender_counts["M"])
    gender_counts["Male proportion"] = gender_counts["M"] / (gender_counts["F"] + gender_counts["M"])

    # Create the figure with Plotly
    fig = go.Figure()

    # Add female directors on top of the male producers bar trace
    fig.add_trace(go.Bar(
        x=gender_counts.index,
        y=gender_counts["Female proportion"],
        name="Female",
        marker_color= color_F,
        opacity=0.8
    ))

    # Add male directors as a bar trace
    fig.add_trace(go.Bar(
        x=gender_counts.index,
        y=gender_counts["Male proportion"],
        name="Male",
        marker_color=color_M,
        opacity=0.8
    ))

    # Update the layout to stack the bars
    fig.update_layout(
        barmode='stack',
        title=f"Proportion of Movies with male and female Directors from {start_year} to {end_year}",
        xaxis_title="Year",
        yaxis_title="Count of Movies",
        legend_title="Director Gender",
        yaxis_tickformat='0%', 
        bargap=0
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_G)

    # Export the plot to an HTML file
    fig.write_html(output_html)

    fig.show()


def visualize_type_of_role_credited(movies_import, gender="B"):
    """
    Visualize the distribution of the type of role credited (Credited by name, job title...) across movies.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".
    gender (str): M, F or B (both)
    Returns:
    None
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    if gender == "B":
        movies = movies_import.copy()
        plt.title(
            "Distribution of the type of role credited across movies for both gender"
        )
    else:
        movies = movies_import[movies_import["actor_gender"] == gender]
        if gender == "M":
            plt.title(
                "Distribution of the type of role credited across movies for actors"
            )
        elif gender == "F":
            plt.title(
                "Distribution of the type of role credited across movies for actresses"
            )
    # Group by decade
    movies["decade"] = (movies["movie_release_date"].dt.year // 10) * 10
    df_cummu = (
        movies.groupby(movies.decade)["role_cat"]
        .value_counts(normalize=True)
        .reset_index()
    )

    df = df_cummu.set_index("decade", inplace=True)
    _ = df_cummu.pivot(columns="role_cat", values="proportion").plot.area(ax=ax)

    _ = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()
    return


def visualize_director_producer_actor_gender_correlation_boxplot(movies):
    """
    Visualize the correlation between the presence of a female director and/or a female producer and the number of female actors in movies using box plots.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies, directors, producers, and actors.

    Returns:
    None
    """

    # Calculate the count of female actors for each movie
    female_actor_counts = movies[movies["actor_gender"] == "F"].groupby("wikiID").size()

    # Create binary columns for the presence of a female director and a female producer
    movies["has_female_director"] = movies["director_gender"] == "F"
    movies["has_female_producer"] = movies["producer_gender"] == "F"

    # Merge the count of female actors into the movies DataFrame
    movies = movies.merge(
        female_actor_counts.rename("female_actor_count"), on="wikiID", how="left"
    )

    # Drop duplicates since there can be multiple actors per movie
    movies.drop_duplicates(subset="wikiID", inplace=True)

    # Create a categorical variable for the four groups
    conditions = [
        (~movies["has_female_director"] & ~movies["has_female_producer"]),
        (~movies["has_female_director"] & movies["has_female_producer"]),
        (movies["has_female_director"] & ~movies["has_female_producer"]),
        (movies["has_female_director"] & movies["has_female_producer"]),
    ]
    choices = [
        "No Female Director/Producer",
        "Female Producer Only",
        "Female Director Only",
        "Both Female Director and Producer",
    ]
    movies["category"] = np.select(conditions, choices)

    # Specify the order for the box plot
    category_order = [
        "No Female Director/Producer",
        "Female Producer Only",
        "Female Director Only",
        "Both Female Director and Producer",
    ]

    plt.figure(figsize=(14, 8))
    sns.set_style("darkgrid")

    # Create a box plot with specified order
    sns.boxplot(
        data=movies,
        x="category",
        y="female_actor_count",
        order=category_order,
        palette="pastel",
    )

    plt.title(
        "Distribution of Female Actors by Presence of Female Directors and Producers"
    )
    plt.xlabel("Category")
    plt.ylabel("Number of Female Actors")
    plt.xticks(rotation=45)  # Rotate the x labels for better readability
    plt.show()


def visualize_proportion_gender_credited(movies):
    """
    Visualize the distribution uncredited/credited roles by gender.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".
    Returns:
    None
    """

    F_credited = movies[(movies["credited"] == True) & (movies["actor_gender"] == "F")][
        "actor_name"
    ].count()
    F_uncredited = movies[
        (movies["credited"] == False) & (movies["actor_gender"] == "F")
    ]["actor_name"].count()
    M_credited = movies[(movies["credited"] == True) & (movies["actor_gender"] == "M")][
        "actor_name"
    ].count()
    M_uncredited = movies[
        (movies["credited"] == False) & (movies["actor_gender"] == "M")
    ]["actor_name"].count()

    fig, ax = plt.subplots()
    ax.pie(
        [F_credited, F_uncredited, M_credited, M_uncredited],
        labels=["F_credited", "F_uncredited", "M_credited", "M_uncredited"],
    )
    plt.show()
    return


def visualize_proportion_specific_gender_credited(movies, gender="F"):
    """
    Visualize the distribution uncredited/credited roles by gender.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".
    Returns:
    None
    """

    X_credited = movies[
        (movies["credited"] == True) & (movies["actor_gender"] == gender)
    ]["actor_name"].count()
    X_uncredited = movies[
        (movies["credited"] == False) & (movies["actor_gender"] == gender)
    ]["actor_name"].count()
    percentage_uncredited = round(X_uncredited / (X_credited + X_uncredited) * 100, 2)
    fig, ax = plt.subplots()
    gender_job = "actors" if gender == "M" else "actresses"
    plt.title(
        f"{percentage_uncredited} of actors are uncredited".format(
            percentage_uncredited=percentage_uncredited
        )
    )
    ax.pie(
        [X_credited, X_uncredited],
        labels=[f"{gender} credited", f"{gender} uncredited"],
    )
    plt.show()
    return


def visualize_un_credited_stacked(movies_import, gender="B"):
    """
    Visualize the distribution of the type of role credited (Credited by name, job title...) across movies.

    Parameters:
    movies (pandas.DataFrame): DataFrame containing information about movies and actors.
    style (str): Style of the plot. Default is "darkgrid".
    gender (str): M, F or B (both)
    Returns:
    None
    """

    fig, ax = plt.subplots(figsize=(9, 6))

    if gender == "B":
        movies = movies_import.copy()
        plt.title("Distribution of whether a role was credited or not")
    else:
        movies = movies_import[movies_import["actor_gender"] == gender]
        if gender == "M":
            plt.title("Distribution of whether a role was credited or not for an actor")
        elif gender == "F":
            plt.title(
                "Distribution of whether a role was credited or not for actresses"
            )
    # Group by decade
    movies["decade"] = (movies["movie_release_date"].dt.year // 10) * 10
    df_cummu = (
        movies.groupby(movies.decade)["credited"]
        .value_counts(normalize=True)
        .reset_index()
    )

    df = df_cummu.set_index("decade", inplace=True)
    _ = df_cummu.pivot(columns="credited", values="proportion").plot.area(ax=ax)

    _ = plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.show()
    return


def visualize_wordcloud_job_roles(movies):
    role_women = movies.loc[movies["actor_gender"] == "F"].copy(deep=True)
    role_women = role_women[role_women["role_cat"] == "JOB"]
    role_women = drop_roles(role_women)

    role_men = movies.loc[movies["actor_gender"] == "M"].copy(deep=True)
    role_men = role_men[role_men["role_cat"] == "JOB"]
    role_men = drop_roles(role_men)

    women_counts = role_women["role_str"].value_counts().head(10)
    men_counts = role_men["role_str"].value_counts().head(10)
    # Create WordClouds for women and men
    women_wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Oranges"
    ).generate_from_frequencies(women_counts)
    men_wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Oranges"
    ).generate_from_frequencies(men_counts)

    # Plotting the WordClouds
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(women_wordcloud, interpolation="bilinear")
    plt.title("Women Jobs as a Characters", color="black", fontsize=16)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(men_wordcloud, interpolation="bilinear")
    plt.title("Men Jobs as a Characters", color="black", fontsize=16)
    plt.axis("off")

    plt.show()
    return


def visualize_wordcloud_r2n_roles(movies):
    role_women = movies.loc[movies["actor_gender"] == "F"].copy(deep=True)
    role_women = role_women[role_women["role_cat"] == "ROLE_TO_NAME"]
    role_women = role_women.drop(
        role_women[role_women["role_to_name"].str.contains("Self")].index
    )
    role_women = role_women.drop(
        role_women[role_women["role_to_name"].str.contains("Narrator")].index
    )
    role_women = role_women.drop(
        role_women[role_women["role_to_name"].str.contains("voice")].index
    )
    #role_women["role_to_name"] = role_women["role_to_name"].apply(lambda x: x.split("'s")[-1])
    role_men = movies.loc[movies["actor_gender"] == "M"].copy(deep=True)
    role_men = role_men[role_men["role_cat"] == "ROLE_TO_NAME"]
    role_men = role_men.drop(role_men[role_men["role_to_name"].str.contains("Self")].index)
    role_men = role_men.drop(role_men[role_men["role_to_name"].str.contains("Narrator")].index)
    role_men = role_men.drop(role_men[role_men["role_to_name"].str.contains("voice")].index)
    #role_men["role"] = role_men["role"].apply(lambda x: x.split("'s")[-1])

    women_counts = role_women["role_to_name"].value_counts().head(10)
    men_counts = role_men["role_to_name"].value_counts().head(10)
    print(women_counts)
    # Create WordClouds for women and men
    women_wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Oranges"
    ).generate_from_frequencies(women_counts)
    men_wordcloud = WordCloud(
        width=800, height=400, background_color="white", colormap="Oranges"
    ).generate_from_frequencies(men_counts)

    # Plotting the WordClouds
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(women_wordcloud, interpolation="bilinear")
    plt.title("Women Supporting Role as Characters", color="black", fontsize=16)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(men_wordcloud, interpolation="bilinear")
    plt.title("Men Supporting Role as Characters", color="black", fontsize=16)
    plt.axis("off")

    plt.show()
    return

def visualize_barplot_r2n_roles(movies):
    compare_Len = 10
    role_women = movies.loc[movies["actor_gender"] == "F"].copy(deep=True)
    role_women = role_women[role_women["role_cat"] == "ROLE_TO_NAME"]
    role_women = drop_roles(role_women)


    role_men = movies.loc[movies["actor_gender"] == "M"].copy(deep=True)
    role_men = role_men[role_men["role_cat"] == "ROLE_TO_NAME"]
    role_men = drop_roles(role_men)

    women_counts = role_women["role_str"].value_counts().head(10)
    men_counts = role_men["role_str"].value_counts().head(10)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    roles = list(women_counts.index)[::-1]
    counts = list(women_counts.values)[::-1]
    plt.barh(roles, counts, color=color_F)
    plt.ylabel('Count')
    plt.ylabel('Role')
    plt.title(f'Top {compare_Len} Women\'s Roles in a relationship to a named character')
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    roles = list(men_counts.index)[::-1]
    counts = list(men_counts.values)[::-1]
    plt.barh(roles, counts, color=color_M)
    plt.xlabel('Count')
    plt.ylabel('Role')
    plt.title(f'Top {compare_Len} Men\'s Roles in a relationship to a named character')
    plt.tight_layout()
    plt.show()

def visualize_barplot_r2n_roles_HTML(movies, output_html='html_plots/r2n_roles_bar.html'):
    compare_Len = 10
    # Filtering and processing roles for women
    role_women = movies.loc[movies["actor_gender"] == "F"].copy(deep=True)
    role_women = role_women[role_women["role_cat"] == "ROLE_TO_NAME"]
    role_women = drop_roles(role_women)

    # Filtering and processing roles for men
    role_men = movies.loc[movies["actor_gender"] == "M"].copy(deep=True)
    role_men = role_men[role_men["role_cat"] == "ROLE_TO_NAME"]
    role_men = drop_roles(role_men)

    # Counting top roles
    women_counts = role_women["role_str"].value_counts().head(compare_Len)
    men_counts = role_men["role_str"].value_counts().head(compare_Len)

    # Creating figure with subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Top {compare_Len} Women\'s Roles', f'Top {compare_Len} Men\'s Roles'))

    # Women's roles plot
    fig.add_trace(go.Bar(
        x=women_counts.values[::-1],
        y=women_counts.index[::-1],
        orientation='h',
        marker_color=color_F,  # Example color, replace with desired color
        name='Women'
    ), row=1, col=1)

    # Men's roles plot
    fig.add_trace(go.Bar(
        x=men_counts.values[::-1],
        y=men_counts.index[::-1],
        orientation='h',
        marker_color=color_M,  # Example color, replace with desired color
        name='Men'
    ), row=1, col=2)

    # Update layout
    fig.update_layout(
        title_text='Top Roles in Relationship to a Named Character',
        yaxis=dict(title='Role'),
        yaxis2=dict(title='Role'),
        xaxis=dict(title='Count'),
        xaxis2=dict(title='Count')
    )

    fig.show()
    fig.write_html(output_html)

def drop_roles(movies):
    movies = movies.drop(
        movies[movies["role_str"].str.contains("self")].index
    )
    movies = movies.drop(
        movies[movies["role_str"].str.contains("narrator")].index
    )
    movies = movies.drop(
        movies[movies["role_str"].str.contains("voice")].index
    )
    movies = movies.drop(
        movies[movies["role_str"]==""].index
    )
    movies = movies.drop(
        movies[movies["role_str"].str.contains("#")].index
    )
    # Drop misclassified named roles
    movies = movies.drop(
        movies[movies["role_str"]=="m"].index
    )
    movies = movies.drop(
        movies[movies["role_str"]=="dot"].index
    )
    movies = movies.drop(
        movies[movies["role_str"]=="q"].index
    )
    movies = movies.drop(
        movies[movies["role_str"]=="candy"].index
    )
    return movies

def visualize_bar_job_roles_bar(movies, YEAR_RANGE=[1980, 2010], compare_Len = 10):
    movies = movies[(movies["year"] >= YEAR_RANGE[0]) & (movies["year"] <= YEAR_RANGE[1])].copy(deep=True)
    role_women = movies.loc[movies["actor_gender"] == "F"].copy(deep=True)
    role_women = role_women[role_women["role_cat"] == "JOB"]
    role_women = drop_roles(role_women)

    role_men = movies.loc[movies["actor_gender"] == "M"].copy(deep=True)
    role_men = role_men[role_men["role_cat"] == "JOB"]
    role_men = drop_roles(role_men)
    women_counts = role_women["role_str"].value_counts().head(compare_Len)
    men_counts = role_men["role_str"].value_counts().head(compare_Len)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    job_roles = list(women_counts.index)[::-1]
    counts = list(women_counts.values)[::-1]
    plt.barh(job_roles, counts, color=color_F)
    plt.xlabel('Count')
    plt.ylabel('Job Role')
    plt.title(f'Top {compare_Len} Women Job Roles')
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.subplot(1, 2, 2)
    job_roles = list(men_counts.index)[::-1]
    counts = list(men_counts.values)[::-1]
    plt.barh(job_roles, counts, color=color_M)
    plt.xlabel('Count')
    plt.ylabel('Job Role')
    plt.title(f'Top {compare_Len} Men Job Roles')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def visualize_bar_job_roles_bar_HTML(movies, YEAR_RANGE=[1980, 2010], compare_Len=10, output_html='html_plots/job_roles_bar.html'):
    """
    Visualizes the top job roles in movies for men and women within a specified year range using Plotly.

    Parameters:
    movies (DataFrame): The dataset containing movie information.
    YEAR_RANGE (list): The range of years to consider.
    compare_Len (int): The number of top roles to compare.
    """

    # Filtering movies by year range
    movies = movies[(movies["year"] >= YEAR_RANGE[0]) & (movies["year"] <= YEAR_RANGE[1])].copy(deep=True)

    # Filtering and processing roles for women
    role_women = movies.loc[movies["actor_gender"] == "F"].copy(deep=True)
    role_women = role_women[role_women["role_cat"] == "JOB"]
    role_women = drop_roles(role_women)

    # Filtering and processing roles for men
    role_men = movies.loc[movies["actor_gender"] == "M"].copy(deep=True)
    role_men = role_men[role_men["role_cat"] == "JOB"]
    role_men = drop_roles(role_men)

    # Getting top roles
    women_counts = role_women["role_str"].value_counts().head(compare_Len)
    men_counts = role_men["role_str"].value_counts().head(compare_Len)

    # Creating subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'Top {compare_Len} Women Job Roles', f'Top {compare_Len} Men Job Roles'))

    # Plot for Women
    fig.add_trace(
        go.Bar(
            x=women_counts.values[::-1],
            y=women_counts.index[::-1],
            orientation='h',
            name='Women'
        ),
        row=1, col=1
    )
    fig.update_traces(marker_color=[color_F]*compare_Len)
    # Plot for Men
    fig.add_trace(
        go.Bar(
            x=men_counts.values[::-1],
            y=men_counts.index[::-1],
            orientation='h',
            name='Men'
        ),
        row=1, col=2
    )
    fig.update_traces(selector=({'name':'Men'}), marker_color=[color_M]*compare_Len)
    fig.update_traces(selector=({'name':'Women'}), marker_color=[color_F]*compare_Len)
    
    # Update layout
    fig.update_layout(title_text="Top Job Roles in Movies")
    fig.show()
    fig.write_html(output_html)

def visualize_prop_of_actor_and_bd_rating(movies):
    sns.set_style("darkgrid")
    movies_agg = movies.copy()

    movies_agg.drop(
        ["Female_Actors_Per_Film", "Male_Actors_Per_Film", "Total_Actors_Per_Film"],
        axis=1,
        inplace=True,
    )

    movies_agg.plot.bar(rot=0, color=[color_F, color_M], figsize=(6, 3), stacked=True)

    plt.title("Proportion of Male and Female Actors for Each Bechdel Test Rating")
    plt.xlabel("Bechdel Test Rating")
    plt.ylabel("Proportion")
    plt.legend(loc="upper right")
    plt.show()


def visualize_prop_of_actor_and_bd_rating_HTML(movies, output_html="html_plots/actors_bdrating_proportion.html"):
    movies_agg = movies.copy()
    movies_agg.drop(
        ["Female_Actors_Per_Film", "Male_Actors_Per_Film", "Total_Actors_Per_Film"],
        axis=1,
        inplace=True,
    )

    # Set color for each gender
    color_discrete_map = {"Female Proportion": color_F, "Male Proportion": color_M}

    # Create a piled histogram using Plotly
    fig = px.bar(
        movies_agg,
        x=movies_agg.index,
        y=movies_agg.columns,
        title="Proportion of male and female Actors for each Bechdel test rating",
        labels={
            "bechdel_rating": "Bechdel test rating",
            "value": "Percentage (%)",
            "variable": "Actor Gender",
        },
        color_discrete_map=color_discrete_map,
        opacity=0.8
    )
    # Update x-axis to treat as category
    fig.update_xaxes(type='category')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=color_G)
    # Update y-axis to show percentages
    fig.update_layout(yaxis_tickformat='0%', bargap=0)
    fig.show()
    fig.write_html(output_html)


def visualize_regression_HTML(movies, output_html="html_plots/regression_actresses_rating.html"):
    """
    Visualize the correlation between the percentage of women actors in movies and their average rating using Plotly.

    Args:
        movies (pandas.DataFrame): A DataFrame containing the percentage of women actors and the average rating of movies.
    """

    # Calculating the percentage of female actors per film
    movies['Percentage_Women_Actors'] = (
        movies['Female_Actors_Per_Film'] / movies['Total_Actors_Per_Film']
    )

    # Creating the scatter plot with a trendline
    fig = px.scatter(
        movies,
        x='Percentage_Women_Actors',
        y='vote_average',
        trendline='ols',
        labels={'vote_average': 'Average Rating', 'Percentage_Women_Actors': 'Percentage of Women Actresses'},
        title='Correlation between Rating and Proportion of Women Actresses',
        color_discrete_sequence=[color_G],
        opacity=0.5
    )

    # Updating the color of the trendline
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.line.color = color_F
            break
    fig.update_layout(xaxis_tickformat='0%') 
    
    # Write in HTML
    fig.show()
    fig.write_html(output_html)



def visualize_popularity(reception_bechdel):
    colors = [color_G, color_B]

    # The x locations for the groups
    ind = range(len(reception_bechdel))

    # The width of the bars
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3))

    for i, col in enumerate(reception_bechdel.columns):
        # Calculate offset for each bar
        offset = (i - 1) * width + width / 2
        ax.bar(
            [x + offset for x in ind],
            reception_bechdel[col],
            width,
            label=col,
            color=colors[i]
        )

    # Add labels
    ax.set_ylabel('Count')
    ax.set_title('Popularity of Movies for each Bechdel test rating')
    ax.set_xticks([x + width/2 for x in ind])
    ax.set_xticklabels(reception_bechdel.index)
    ax.set_xlabel('Bechdel Test Rating')
    ax.legend()
    plt.show()

# Remember to define color_G and color_B before calling this function.



def visualize_popularity_HTML(reception_bechdel, output_html="html_plots/popularity.html"):
    fig = px.bar(
        reception_bechdel,
        x=reception_bechdel.index,
        y=reception_bechdel.columns,
        barmode="group",
        title="Popularity of Movies for each Bechdel test rating",
        labels={
            "index": "Bechdel Test rating",
            "variable": "Metric",
            "popularity": "Popularity",
            "vote_average": "Average Rating"
        },
        color_discrete_map={reception_bechdel.columns[0]: color_G, reception_bechdel.columns[1]: color_B},
        opacity=0.8
    )

    # Update layout
    fig.update_layout(
        xaxis_title="Bechdel Test Rating",
        yaxis_title="Count",
    )

    fig.show()

    # Save the figure as HTML
    fig.write_html(output_html)


def visualize_bd_rating_evolution(yearly_bechdel):
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=yearly_bechdel, x="release_year", y="proportion_passing")
    plt.title("Proportion of Movies Passing the Bechdel Test Over Time")
    plt.xlabel("Year")
    plt.ylabel("Proportion Passing")
    plt.grid(True)
    plt.show()

def visualize_bd_rating_evolution_HTML(yearly_bechdel, year_range=[], output_html="html_plots/bd_rating_evolution.html"):
    fig = px.scatter(yearly_bechdel,
                  y='proportion_passing', 
                  trendline='ols',
                  title=f"Proportion of movies passing the Bechdel test between {year_range[0]} and {year_range[1]}",
                  labels={'release_year': 'Year', 'proportion_passing': 'Proportion passing the Bechdel test'},
                  color_discrete_sequence=[color_G],
                  )
    fig.update_layout(xaxis_title='Year', yaxis_title='Proportion passing the Bechdel test', yaxis_tickformat='0%', template='plotly_white')
    fig.update_xaxes(range=year_range)
    fig.update_yaxes(range=[0, 1])

    # Updating the color of the trendline
    for trace in fig.data:
        if trace.mode == 'lines':
            trace.line.color = color_F
            break

    fig.show()
    fig.write_html(output_html)


def visualize_number_of_movies(yearly_bechdel):
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=yearly_bechdel, x="release_year", y="total_movies")
    plt.title("Total Number of Movies Produced Over Time")
    plt.xlabel("Year")
    plt.ylabel("Total Movies Produced")
    plt.grid(True)
    plt.show()


def visualize_genres_dist_on_bd(grouped_data):
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 10))
    sns.barplot(
        x="Proportion_Male",
        y="main_genre",
        data=grouped_data.head(15),
        color=color_M,
        label="Male",
        alpha=0.8,
    )

    sns.barplot(
        x="Proportion_Female",
        y="main_genre",
        data=grouped_data.head(15),
        color=color_F,
        label="Female",
        alpha=0.8,
    )
    
    plt.legend()
    plt.title("Proportion of Male and Female Actors in Top 10 Movie Genres")
    plt.xlabel("Proportion")
    plt.ylabel("Genres")
    plt.show()
    return


def visualize_genres_dist_on_bd_HTML(grouped_data, output_html="html_plots/bd_genre_ds.html"):
    top_genres = grouped_data.head(15)

    melted_data = top_genres.melt(
        id_vars="main_genre",
        value_vars=["Proportion_Male", "Proportion_Female"],
        var_name="Gender",
        value_name="Proportion",
    )
    melted_data = melted_data.sort_values(by='Proportion', ascending=False)

    fig = px.bar(
        melted_data,
        x="Proportion",
        y="main_genre",
        color="Gender",
        orientation="h",
        barmode="stack",
        color_discrete_map={
            "Proportion_Male": color_M,
            "Proportion_Female": color_F,
        },
        labels={"main_genre": "Genres"},
        title="Proportion of Male and Female Actors in Top 15 Movie Genres",
        opacity=0.8
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=color_G)

    fig.update_layout(bargap=0, xaxis_tickformat='0%')
    fig.show()
    fig.write_html(output_html)

def visualize_number_of_movies_HTML(yearly_bechdel, year_range=[], output_html="html_plots/number_of_movies_bechdel.html"):
    fig = px.scatter(yearly_bechdel,
                y='total_movies',
                title=f"Number of movies produced per year which are in the Bechdel dataset between {year_range[0]} and {year_range[1]}",
                labels={'release_year': 'Year', 'total_movies': 'Movies produced per year'},
                color_discrete_sequence=[color_G],
                opacity=0.6
                )
                
    fig.update_layout(xaxis_title='Year', 
                      yaxis_title='Total Movies Produced',
                      showlegend=False, 
                      template='plotly_white')
    fig.update_xaxes(range=year_range)
    fig.show()
    fig.write_html(output_html)

def visualize_bar_plot_role_cat(movies, YEAR_RANGE=[1980, 2010]):
    fig, axs = plt.subplots(1, 5, figsize=(12, 2))
    movies = movies[(movies['year']>=YEAR_RANGE[0]) & (movies['year']<=YEAR_RANGE[1])].copy(deep=True)
    # Plot data in each subplot
    # All actors
    axs[0].bar('Actors', movies.groupby('actor_gender')['actor_name'].count()['M'], color=color_M)
    axs[0].bar('Actress', movies.groupby('actor_gender')['actor_name'].count()['F'], color=color_F)
    axs[0].set_ylabel('Number of roles')
    axs[0].set_title('All roles')
    # Named characters
    axs[1].bar('Actors', movies[movies['role_cat']=='NAME'].groupby('actor_gender')['actor_name'].count()['M'], color=color_M)
    axs[1].bar('Actress', movies[movies['role_cat']=='NAME'].groupby('actor_gender')['actor_name'].count()['F'], color=color_F)
    axs[1].set_ylabel('Number of roles')
    axs[1].set_title('Named roles')
    # Characters with jobs
    axs[2].bar('Actors', movies[movies['role_cat']=='JOB'].groupby('actor_gender')['actor_name'].count()['M'], color=color_M)
    axs[2].bar('Actress', movies[movies['role_cat']=='JOB'].groupby('actor_gender')['actor_name'].count()['F'], color=color_F)
    axs[2].set_ylabel('Number of roles')
    axs[2].set_title('Job roles')
    # Characters in relation with a named character
    axs[3].bar('Actors', movies[movies['role_cat']=='ROLE_TO_NAME'].groupby('actor_gender')['actor_name'].count()['M'], color=color_M)
    axs[3].bar('Actress', movies[movies['role_cat']=='ROLE_TO_NAME'].groupby('actor_gender')['actor_name'].count()['F'], color=color_F)
    axs[3].set_ylabel('Number of roles')
    axs[3].set_title('Relation to named role')
    # Characters in relation with a named character
    axs[4].bar('Actors', movies[movies['role_cat']=='ROLE_TO_JOB'].groupby('actor_gender')['actor_name'].count()['M'], color=color_M)
    axs[4].bar('Actress', movies[movies['role_cat']=='ROLE_TO_JOB'].groupby('actor_gender')['actor_name'].count()['F'], color=color_F)
    axs[4].set_ylabel('Number of roles')
    axs[4].set_title('Relation to job role')

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=1)

    # Show the plot
    plt.show()

def visualize_bar_plot_role_cat_HTML(movies, YEAR_RANGE=[1980, 2010], output_html="html_plots/bar_plt_role_cat.html"):
    movies_filtered = movies[(movies['year'] >= YEAR_RANGE[0]) & (movies['year'] <= YEAR_RANGE[1])]

    # Create subplots
    fig = make_subplots(rows=1, cols=5, subplot_titles=('All roles', 'Named roles', 'Job roles', 
                                                         'Relation to named role', 'Relation to job role'))

    # Categories for the plots
    categories = ['All', 'NAME', 'JOB', 'ROLE_TO_NAME', 'ROLE_TO_JOB']

    for i, cat in enumerate(categories, start=1):
        if cat == 'All':
            df_filtered = movies_filtered
        else:
            df_filtered = movies_filtered[movies_filtered['role_cat'] == cat]

        count_M = df_filtered[df_filtered['actor_gender'] == 'M']['actor_name'].count()
        count_F = df_filtered[df_filtered['actor_gender'] == 'F']['actor_name'].count()

        fig.add_trace(go.Bar(name='Actors', x=['Actors'], y=[count_M], marker_color=color_M), row=1, col=i)
        fig.add_trace(go.Bar(name='Actresses', x=['Actresses'], y=[count_F], marker_color=color_F), row=1, col=i)

    # Update layout
    fig.update_layout(showlegend=False, title_text="Number of Roles by Category and Gender")
    fig.show()
    fig.write_html(output_html)


def role_job_barplot_history(movies, job, YEAR_RANGE=[1980, 2010]):
    movies = movies[(movies['year']>=YEAR_RANGE[0]) & (movies['year']<=YEAR_RANGE[1])].copy(deep=True)
    if type(job) is list:
        subset = pd.DataFrame(movies[(movies['role_cat']=="JOB") & ((movies['role_str']==job[0]) | (movies['role_str']==job[1]))].groupby([movies['year'] // 10 * 10, 'actor_gender']).count().iloc[:,0]).reset_index()
    else:
        subset = pd.DataFrame(movies[(movies['role_cat']=="JOB") & (movies['role_str']==job)].groupby([movies['year'] // 10 * 10, 'actor_gender']).count().iloc[:,0]).reset_index()
    # Pivot the data to get counts by year and gender
    pivot_df = subset.pivot_table(index='year', columns='actor_gender', values='actor_name', fill_value=0)
    # Normalize the data to get proportions
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    # Create a stacked bar plot
    pivot_df.plot(kind='bar', stacked=True, figsize=(10, 2), color=[color_F, color_M])
    plt.xlabel('Decade')
    plt.ylabel('Percentage')
    if type(job) is list:
        plt.title(f'Gender distribution of actors credited as {job[0]}-{job[1]}, by decade')
    else:
        plt.title(f'Gender distribution of actors credited as {job}, by decade')
    plt.show()

def role_job_barplot_history_HTML(movies, job, YEAR_RANGE=[1980, 2010]):
    output_html=f"html_plots/role_job_barplot_history_{job}.html"
    movies = movies[(movies['year'] >= YEAR_RANGE[0]) & (movies['year'] <= YEAR_RANGE[1])].copy(deep=True)

    # Grouping and counting based on job roles and gender
    if isinstance(job, list):
        subset = pd.DataFrame(movies[(movies['role_cat'] == "JOB") & ((movies['role_str'] == job[0]) | (movies['role_str'] == job[1]))].groupby([movies['year'] // 10 * 10, 'actor_gender']).count().iloc[:, 0]).reset_index()
    else:
        subset = pd.DataFrame(movies[(movies['role_cat'] == "JOB") & (movies['role_str'] == job)].groupby([movies['year'] // 10 * 10, 'actor_gender']).count().iloc[:, 0]).reset_index()

    # Pivot the data to get counts by year and gender
    pivot_df = subset.pivot_table(index='year', columns='actor_gender', values='actor_name', fill_value=0)

    # Normalize the data to get proportions
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    # Creating the bar plot
    fig = go.Figure()
    for gender in pivot_df.columns:
        fig.add_trace(go.Bar(
            x=pivot_df.index,
            y=pivot_df[gender],
            name=gender,
            text=pivot_df[gender],
            textposition='auto',
        ))

    fig.update_traces(selector={'name': 'M'}, marker_color=color_M)
    fig.update_traces(selector={'name': 'F'}, marker_color=color_F)

    # Update layout for stacked bar plot
    fig.update_layout(
        barmode='stack',
        title=f'Gender Distribution of Actors Credited as {job}, by Decade',
        xaxis_title='Decade',
        yaxis_title='Percentage',
        yaxis=dict(tickformat=".0%")
    )

    fig.show()
    fig.write_html(output_html)



def visualize_box_plot_role_avg_pay(movies, YEAR_RANGE):
    movies = movies[(movies['year']>=YEAR_RANGE[0]) & (movies['year']<=YEAR_RANGE[1])].copy(deep=True)
    plt.figure(figsize=(10, 6))
    plt.title('Average pay for the job played by actors and actresses')
    sns.boxplot(data=[movies[(movies['actor_gender']=="M") & (movies['avg_pay'].notna())]['avg_pay']/12, movies[(movies['actor_gender']=="F") & (movies['avg_pay'].notna())]['avg_pay']/12], palette=[color_M, color_F])
    plt.xticks([0, 1], ['Actor Role', 'Actress Role']) 
    plt.ylabel('Average monthly pay')

def visualize_box_plot_role_avg_pay_HTML(movies, YEAR_RANGE, output_html="html_plots/box_plot_role_avg_pay.html"):
    # Filtering movies by year range
    movies = movies[(movies['year'] >= YEAR_RANGE[0]) & (movies['year'] <= YEAR_RANGE[1])].copy(deep=True)

    # Extracting average pay data for actors and actresses
    actor_pay = movies[(movies['actor_gender'] == "M") & (movies['avg_pay'].notna())]['avg_pay'] / 12
    actress_pay = movies[(movies['actor_gender'] == "F") & (movies['avg_pay'].notna())]['avg_pay'] / 12

    # Creating box plots
    fig = go.Figure()

    fig.add_trace(go.Box(
        y=actor_pay,
        name='Actor Role',
        marker_color=color_M  # Example color, replace with desired color
    ))

    fig.add_trace(go.Box(
        y=actress_pay,
        name='Actress Role',
        marker_color=color_F  # Example color, replace with desired color
    ))

    # Update layout
    fig.update_layout(
        title='Average Pay for the Job Played by Actors and Actresses',
        yaxis_title='Average Monthly Pay',
        xaxis=dict(
            title='Role',
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Actor Role', 'Actress Role']
        )
    )

    fig.show()
    fig.write_html(output_html)