# Are women in movies clickbate?

<div>
  <div><b>Team</b>: Adaptateurs</div>
  <div><b>Supervisor</b>: <a href="https://people.epfl.ch/silin.gao?lang=en"> Silin Gao (PhD) </a> </div>
</div>

<span align="center">

<br>

**Lena Vogel,  François Dumoncel,  Aymeric Bacuet,  Kenji Tetard,  Naël Dillenbourg**

*École Polytechnique Fédérale de Lausanne*

<br> 

</span>

## How to run our Notebooks  
<details>
<summary> Tutorial </summary>

Install necessary package using

```console
$ pip install -r requirements.txt
```

Decompress data just after cloning this repo
1. CMU original dataset 
```console
$ cd data/ && tar -xvzf *.tar.gz
```

1. TMDB 
```console
$ cd external && unzip movies.zip && mv "Movies Dataset" Movies
```

Or simply decompress archive from file system. You can also directly use the pre-processed pickle file in `data/Processed`
</details>

## Repository structure

```
├─── data
│    ├─── external
│           └─── movies 
│    ├─── Processed
│    ├─── MovieSummaries
├─── analysis 
│      └─── visualizer.py
├─── preprocessing
       ├─── loader.py
       ├─── cleaner.py
       └─── merger.py
├─── nlp  
       ├─── nlp_utils.py
       └─── female_keywords.txt
├─── P2.ipynb
├─── P3.ipynb
├─── requirements.txt
├─── Makefile
└─── README.md
```

## Table of Contents 📕

<p>
  <a href="#abstract-"> 1. Abstract</a> 
  <br>
  <a href="#research-questions-">2. Research Questions</a> 
  <br>
  <a href="#datasets-">3. Additional Datasets</a> 
  <br>
  <a href="#methods-">4. Methods</a> 
  <br>
  <a href="#team-organization-">6. Team Organization</a>
</p>

## Abstract 📌

<span align="justify">
In this data analysis, we will study how women are represented in movies, why, and what the impact of their presence is. To do so, we will use the ratio of actors and actresses as the main metric. We will analyze the quality of the feminine characters in movies, such as whether they have a name. Furthermore, we will try to understand the cause of the over- or sub-representation of women characters, and the effect it has on the story.
To do so, we will use the movies contained in the CMU Summary corpus, along with the genders of the movies' crews scrapped from Wikidata and the roles of the characters taken from TMDB.
In order to complete our quality evalutation of the representation of women, we will use the results of the Bechdel test, which evaluates the importance of the feminine characters in a movie.
This study aims to provide a comprehensive analysis of female representation in cinema, using a data-driven approach to evaluate how gender dynamics in movies influence both narrative quality and societal perceptions.

</span>

## Research Questions 🔍

Our journey through the cinematic universe is guided by the following pivotal questions:

- How do various film industries, such as Hollywood, Bollywood, and European cinema, portray women in their films?
- How does the representation of women, both in front of the camera and behind the scenes, shape a film's reception and resonance with its audience?
- Can the Bechdel test serve as a meaningful barometer for evaluating a film's portrayal of women, beyond its surface-level criteria?
- In what ways do film summaries, as condensed narratives, reflect gender biases or empowerment, influenced by the gender dynamics within the film's creation?
- Which genres in cinema are leading the charge in representing women as complex, multi-dimensional characters?

## Datasets 📊

In this analysis, we will use three different datasets

- The original and provided [CMU](http://www.cs.cmu.edu/~ark/personas/) dataset.
- The [TMDB Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset): this dataset offers metadata for over 45,000 movies, with a particular focus on user ratings. Our primary interest lies in getting the populariy, the number of votes each movie receives and its average score rating to define a success metric. This dataset also includes valuable details about a movie's crew, such as the gender of the director and producers. These data points are crucial for conducting analyses on women's roles behind the camera.
- The [Bechdel Dataset](https://bechdeltest.com) that contains the Bechdel rating score for more than 10'000 movies. This dataset will be used to conduct a analysis on movies that passes (or not) the Bechdel test.
- [Maybe Oscar ?]

## Methods 💡

### Preprocessing on **data**

<span align="justify"> 

**Original Dataset**: Initial cleaning has been conducted.

**Bechdel Dataset**: The Bechdel dataset boasts the advantage of minimal missing values, which simplifies the cleaning process significantly.

**TMDB Dataset**: Our focus with the TMDB dataset is to extract data on the crew members behind the scenes, particularly their gender. We mitigate the impact of missing values by employing the genderguesser package, which allows us to infer genders from first names and reduce the volume of incomplete data.

</span>

### Analysis of Women's place in movies


### Techniques
- **Natural Language Processing:** NLP techniques will be applied to film summaries to detect and analyze gender stereotypes.

- **Comparative Analysis:** We'll compare genres, cultures, and production types (independent vs. major studios) in terms of gender representation.

- **Temporal Analysis:** Examining how trends have evolved over time, we'll use this data to potentially forecast future trends.

- **Impact Assessment:** We'll assess the real-world impact of films that pass the Bechdel, particularly in terms of influencing public discourse and contributing to gender equality movements.

## Proposed Timeline 📆

```C
├── Week 8  - Preprocessing
│  
├── Week 9  - Analysis and pipeline
│  
├── Week 10 - Homework II
│  
├── Week 11 - Data Story and fine-grained analysis through Bechdel test and cie
│  
├── Week 12 - Data Story and NLP on plot summaries 
│    
├── Week 13 - Start the website (Github Pages)
│  
├── Week 14 - Finalization
```

## Team Organization ⚙️


| Name            | Email                                | Task                     |
|-----------------|--------------------------------------|--------------------------|
| F. Dumoncel     | `francois.dumoncel-kessler@epfl.ch`  | Preprocessing, Analysis, Clean the repo, README |
| K. Tetard       | `kenji.tetard@epfl.ch`               |  NLP |
| L. Vogel        | `lena.vogel@epfl.ch`                 | README, cleaning, team organization, Bechdel test |
| N. Dillenbourg  | `nael.dillenbourg@epfl.ch`           | Preprocessing, Wikidata scraping   |
| A. Bacuet       | `aymeric.bacuet@epfl.ch`             | Exploratory data Analysis   |
