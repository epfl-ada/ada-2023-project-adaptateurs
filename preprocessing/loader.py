import pandas as pd
import requests
import sys

DATA_PATH = "./data/MovieSummaries/"
EXTERNAL_DATA_PATH = "./data/external/Movies/"


def load_CMU_dataset():
    """
    Loads the CMU dataset from the specified file paths and returns two dataframes:
    character_df and movie_df.

    Returns:
    character_df (pandas.DataFrame), movie_df.
    """

    character_df = pd.read_csv(
        DATA_PATH + "character.metadata.tsv",
        sep="\t",
        header=None,
        names=[
            "wikiID",
            "freebaseID",
            "movie_release_date",
            "character_name",
            "actor_date_of_birth",
            "actor_gender",
            "actor_height_meters",
            "actor_ethni_fbid",
            "actor_name",
            "actor_age_at_movie_release",
            "fbid_char_actor_map",
            "fbid_char",
            "fbid_actor",
        ],
    )

    movie_df = pd.read_csv(
        DATA_PATH + "movie.metadata.tsv",
        sep="\t",
        header=None,
        names=[
            "wikiID",
            "freebaseID",
            "movie_title",
            "movie_release_date",
            "movie_bo_revenue",
            "movie_runtime",
            "fbid_languages",
            "fbid_countries",
            "fbid_genres",
        ],
    )
    return character_df, movie_df


def load_bechdel_dataset():
    """
    Load the Bechdel dataset from the Bechdel Test API.

    Returns:
    bechdel_dataset (pandas.DataFrame): The Bechdel dataset.
    """

    BASE_URL = "http://bechdeltest.com/api/v1/"
    METHOD = "getAllMovies"

    url = f"{BASE_URL}{METHOD}"
    response = requests.get(url, params={})

    if response.status_code == 200:
        data = response.json()

        # Convert to pandas DataFrame
        bechdel_dataset = pd.DataFrame.from_dict(data)
    else:
        print(
            "[ERROR]: Could not retrieve data from API for method: getAllMovies",
            file=sys.stderr,
        )

    return bechdel_dataset


def load_summaries():
    """
    Load plot summaries from a csv file.

    Returns:
    pandas.DataFrame: A dataframe containing the wikiID and summary columns.
    """

    return pd.read_csv(
        DATA_PATH + "plot_summaries.txt",
        sep="\t",
        header=None,
        names=["wikiID", "summary"],
    )


def load_tmdb_dataset():
    """
    Loads the TMDb dataset from the external data path.

    Returns:
    credits_df (pandas.DataFrame): DataFrame containing the credits information.
    meta_df (pandas.DataFrame): DataFrame containing the metadata information.
    """

    credits_df = pd.read_csv(EXTERNAL_DATA_PATH + "credits.csv")
    meta_df = pd.read_csv(EXTERNAL_DATA_PATH + "movies_metadata.csv", low_memory=False)
    return credits_df, meta_df
