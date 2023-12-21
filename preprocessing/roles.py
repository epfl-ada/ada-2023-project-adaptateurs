import pandas as pd
import requests
import spacy
from spacy.matcher import Matcher
# python -m spacy download en_core_web_sm is needed
nlp = spacy.load("en_core_web_trf")

def scrape_roles(movies, API_TOKEN):
    #Define request settings
    headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
    }
    #Add column role to movies
    movies['role']= ""
    for movie_id_db in movies['id'].unique():
        movie_title = movies[movies['id'] == movie_id_db]['movie_title'].values[0]
        movie_year = movies[movies['id'] == movie_id_db]['movie_release_date'].values[0]
        movie_title_url = movie_title.replace(' ', '+')
        url = f"https://api.themoviedb.org/3/search/movie?&query={movie_title_url}"#get movie id
        response = requests.get(url, headers=headers)
        try:
            for movie in response.json()['results']:
                if movie['title'] == movie_title and movie['release_date'].split('-')[0] == str(movie_year).split('-')[0]: #check if movie title and year match
                    movie_id = movie['id']
                    url = f'https://api.themoviedb.org/3/movie/{movie_id}/credits?language=en-US'
                    movie_res = requests.get(url, headers=headers)
                    try:
                        for cast in movie_res.json()['cast']:
                            movies.loc[(movies['id']==movie_id_db) & (movies['actor_name']==cast['name']),'role'] = cast['character'] #add role to movies if corresponding actor
                    except:
                        print('error cast')
        except:
            print('error movie')
    return movies

def role2cat(role):
    if role == 'self':
        return 'self'
    matcher = Matcher(nlp.vocab)
    #Match for role to name (ex: "John's wife")
    pattern = [{"POS": "PROPN", "OP": "+"},
            {"TEXT": "'s"},    
            {"POS": "NOUN"}] 
    matcher.add("ROLE_TO_NAME", [pattern],)
    #Match for job (ex: "Doctor")
    pattern = [{"POS": "NOUN"}] 
    matcher.add("JOB", [pattern])
    #Match for role to job (ex: "Doctor's wife")
    pattern = [{"POS": "NOUN", "OP":"+"},{"TEXT": "'s"},{"POS": "NOUN", "OP": "+"}] # Lowercase match for "wife"
    matcher.add("ROLE_TO_JOB", [pattern])
    #Match for name (ex: "John")
    pattern = [{"POS": "PROPN"}]
    matcher.add("NAME", [pattern])
    doc = nlp(role)
    matches = matcher(doc)
    matched = [nlp.vocab.strings[match_id] for match_id, start, end in matches]
    #Select the best match
    if "ROLE_TO_NAME" in matched:
        return "ROLE_TO_NAME"
    elif "ROLE_TO_JOB" in matched:
        return "ROLE_TO_JOB"
    elif "JOB" in matched:
        return "JOB"
    elif "NAME" in matched:
        return "NAME"
    return "OTHER"

def cleandf(dataset):
    dataset['credited']=True
    dataset['voice_only']=False
    dataset.loc[dataset['role'].str.contains('self'),"role"] = "self"
    dataset.loc[dataset['role'].str.contains('uncredited'),"credited"] = False
    dataset.loc[dataset['role'].str.contains('voice'),"voice_only"] = True
    dataset.loc[dataset['role'].str.contains('Narrator '),"voice_only"] = True
    dataset[dataset['role'].str=='','credited']=False
    return dataset

def get_roles(movies, API_TOKEN):
    movies = scrape_roles(movies, API_TOKEN)
    movies['role_cat'] = movies['role'].apply(role2cat)
    cleandf(movies)
    return movies

def job_comparison(movies, job, YEAR_RANGE=[1980, 2010]):
    movies = movies[(movies['year'] > YEAR_RANGE[0]) & (movies['year'] < YEAR_RANGE[1])].copy(deep=True)
    if type(job) is str:
        subset = movies[(movies['role_cat']=="JOB") & (movies['role_str']==job)]
        total = subset.count().iloc[0]
        subset_F = subset[subset['actor_gender']=="F"].count().iloc[0]
        subset_M = subset[subset['actor_gender']=="M"].count().iloc[0]
        subset_unkown = subset[subset['actor_gender'].isna()].count().iloc[0]
        subset_M = round(subset_M/total, 2)
        subset_F = round(subset_F/total, 2)
        subset_unkown = round(subset_unkown/total, 2)
        print(f"For the role {job}, the actors are {subset_M} men, {subset_F} women and {subset_unkown} unknown. Number of roles {total}.")
    if type(job) is list: #needs to be masculine gender title first, then feminine
        subset0 = movies[(movies['role_cat']=="JOB") & (movies['role_str']==job[0])]
        subset1 = movies[(movies['role_cat']=="JOB") & (movies['role_str']==job[1])]
        total0 = subset0.count().iloc[0]
        total1 = subset1.count().iloc[0]
        totalt = total0 + total1
        subset_M = round(total0/totalt, 2)
        subset_F = round(total1/totalt, 2)
        print(f"For the role {job[0]}-{job[1]}, the actors are {subset_M} men, {subset_F} women. Number of roles {totalt}.")
