import pandas as pd
import requests
import sys

DATA_PATH = "./data/MovieSummaries/"
EXTERNAL_DATA_PATH = "./data/external/Movies Dataset/"


def load_CMU_dataset():
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
    return pd.read_csv(
        DATA_PATH + "plot_summaries.txt",
        sep="\t",
        header=None,
        names=["wikiID", "summary"],
    )


def load_tmdb_dataset():
    credits_df = pd.read_csv(EXTERNAL_DATA_PATH + "credits.csv")
    meta_df = pd.read_csv(EXTERNAL_DATA_PATH + "movies_metadata.csv", low_memory=False)
    return credits_df, meta_df
