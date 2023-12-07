import pandas as pd
import requests
import spacy
from spacy.matcher import Matcher

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
