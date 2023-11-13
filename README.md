# Women's Representation Analysis in the Film Industry

<div>
  <div><b>Team</b>: Adaptateurs</div>
  <div><b>Supervisor</b>: Sillin Gao</div>
</div>

<span align="center">

<br>

**Léna Vogel,  François Dumoncel,  Aymeric Bacuet,  Kenji Tetard,  Naël Dillenbourg**

*École Polytechnique Fédérale de Lausanne*

<br> 

</span>


## How to run our Notebooks

Install necessary package using 

```console
$ pip install -r requirements.txt
```

Decompress data just after cloning this repo
1. CMU original dataset 
```console
$ cd data/ && tar -xvzf *.tar.gz
```

2. TMDB 
```console
$ cd external && unzip movies.zip && mv "Movies Dataset" Movies
```

Or simply decompress archive from file system. You can also directly use the pre-processed pickle file in `data/Processed`

## Reposository structure

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
├─── P2.ipynb
├─── requirements.txt
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
  <a href="#proposed-timeline-">5. Timeline</a> 
  <br>
  <a href="#team-organization-">6. Team Organization</a>
</p>





## Abstract 📌

<span align="justify">

This analysis aims to discover the portrayal of women in the film industry by analyzing a rich dataset encompassing aspects from actor profiles to production details. We aim to uncover gender disparities by examining the roles and ages of actresses at the time of a movie's release, and the influence of female directors and producers on a film's success and reception. We also want to inspect how the Bechdel Test can influence the success of a movie. The impact of female directors and producers on a film's popularity and box-office performance will be examined. Additionally, we will utilize Natural Language Processing technique to assess gender stereotypes within film summaries

</span>

## Research Questions 🔍

- How have the trends in passing the Bechdel and Mako-Mori tests changed over the past few decades?
- Is there a correlation between a movie passing these tests and its commercial success or critical acclaim?
- What genres are more likely to pass or fail these tests, and what does this suggest about genre-specific representation?
- How does the representation of women in films from different countries or cultures fare when applied to these tests?
- What are the common narrative roles or character archetypes for women in films that pass these tests compared to those that don't?
- Are films that pass the Bechdel and Mako-Mori tests also likely to have more gender diversity among their writers, directors, and producers?
- Does the presence of a leading female character or a strong supporting female cast correlate with passing these tests?
- How do independent films compare to major studio productions in terms of passing these tests?
- To what extent do films that pass the Bechdel and Mako-Mori tests influence public discourse on gender representation?
- Are movies that pass these tests more likely to feature women in non-traditional roles or challenge gender stereotypes?
- How does the age rating of a film (G, PG, PG-13, R) impact its likelihood of passing these tests?
- What is the role of film festivals and awards in promoting films that pass the Bechdel and Mako-Mori tests?
- How have audience perceptions and demands influenced the number of films passing these tests over time?
- Can the data from these tests be used to predict future trends in women's cinematic representation?
- What impact, if any, do films that pass these tests have on the real-world issues of gender equality and female empowerment?

## Datasets 📊

In this analysis, we will use three different datasets

1. The original and provided CMU dataset
2. The TMDB Dataset
   - How we got it: directly downloaded from [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) (TMDB)
   - Contains metadata on over 45,000 movies. 26 million ratings from over 270,000 users.
   - Will be used to define a metric for the success of a movie.
3. The Bechdel Dataset
    - How we got it: Obtained via bechdeltest.com API
    - Contains the Bechdel rating score for more than 10 000 movies  
    - See `preprocessing/loader.py/load_bechdel_dataset()`

## Methods 📝

### Preprocessing

<span align="justify">

**Original Dataset**: Initial cleaning has been conducted.

**Bechdel Dataset**: The Bechdel dataset boasts the advantage of minimal missing values, which simplifies the cleaning process significantly.

**TMDB Dataset**: Our focus with the TMDB dataset was to extract data on the crew members behind the scenes, particularly their gender. We mitigate the impact of missing values by employing the genderguesser package, which allows us to infer genders and reduce the volume of incomplete data.

</span>

### Analysis of Women's place

### Bechdel Test

### NLP on plot summaries

## Proposed Timeline 📆


```C
├── Week 8  - Preprocessing
│  
├── Week 9  - Analysis and pipeline
│  
├── Week 10 - Homework II
│  
├── Week 11 - Data Story and fine-grained analysis
│  
├── Week 12 - Data Story and NLP on plot summaries 
│    
├── Week 13 - Start the website (Github Pages)
│  
├── Week 14 - Finalization
```

## Team Organization ⚙️


| Name          | Email                                 | Task                    |
|-----------------|---------------------------------------|----------------------------|
| F. Dumoncel     | `francois.dumoncel-kessler@epfl.ch`     | Preprocessing and README  |
| K. Tetard       | `kenji.tetard@epfl.ch`                  | -   |
| L. Vogel        | `lena.vogel@epfl.ch`                    | -   |
| N. Dillenbourg  | `nael.dillenbourg@epfl.ch`              | -   |
| A. Bacuet       | `aymeric.bacuet@epfl.ch`              | Exploratory and data Analysis   |
