import pandas as pd


def clean_movie_df(movie_df): 
    movie_df['movie_release_date'] = pd.to_datetime(movie_df['movie_release_date'], errors='coerce')
    movie_df['year'] = movie_df['movie_release_date'].dt.year
    return movie_df


def clean_bechdel_df(bechdel_df): 
    bechdel_df = bechdel_df.drop(columns=['id'])
    bechdel_df = bechdel_df.rename(columns={'title': 'movie_title', 'rating' : 'bechdel_rating'})
    return bechdel_df