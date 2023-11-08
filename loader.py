import pandas as pd
import requests
import sys

def load_CMU_dataset():
    character_df = pd.read_csv(
        "data/MovieSummaries/character.metadata.tsv",
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
            "Freebase character/actor map ID",
            "Freebase character ID",
            "Freebase actor ID",
        ],
    )

    movie_df = pd.read_csv(
        "data/MovieSummaries/movie.metadata.tsv",
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
        print("[ERROR]: Could not retrieve data from API for method: getAllMovies", file=sys.stderr)

    return bechdel_dataset