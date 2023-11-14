# Women's Representation Analysis in the Film Industry

<div>
  <div><b>Team</b>: Adaptateurs</div>
  <div><b>Supervisor</b>: <a href="https://people.epfl.ch/silin.gao?lang=en"> Silin Gao (PhD) </a> </div>
</div>

<span align="center">

<br>

**Léna Vogel,  François Dumoncel,  Aymeric Bacuet,  Kenji Tetard,  Naël Dillenbourg**

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

In the contemporary film industry, the representation and portrayal of women have been pivotal topics, sparking widespread discussions about gender disparities. This project aims to delve into these aspects by analyzing a comprehensive dataset that encompasses a wide range of elements from actor profiles to intricate production details. The primary focus is to unearth patterns and trends related to gender disparities, particularly regarding the roles and ages of actresses at the time of a movie's release, as well as the impact of female directors and producers on a film's success and public reception.

The original aspect of our investigation revolves around the Bechdel Test and the Mako-Mori test, which serve as benchmarks for evaluating the representation of women in films. We aim to explore the correlation between a movie passing these tests and its commercial success or critical acclaim. Furthermore, we plan to apply Natural Language Processing (NLP) techniques to assess gender stereotypes within film summaries, offering a nuanced view of how gender roles are depicted in cinema.

</span>

## Research Questions 🔍
Our research is structured around several key questions:

- **Trends Over Time:** How have trends in passing the Bechdel and Mako-Mori tests evolved over the past decades?

**Commercial Success and Critical Acclaim:** Is there a notable correlation between a movie passing these tests and its commercial success or critical acclaim?

**Genre-Specific Representation:** What genres are more likely to pass or fail these tests, and what does this imply about the representation within specific genres?

**Cultural and International Perspectives:** How does women's representation in films from different countries or cultures measure up against these tests?

**Narrative Roles and Character Archetypes:** What common narrative roles or character archetypes for women are observed in films that pass these tests compared to those that don't?

**Gender Diversity in Film Production:** Are films that pass the Bechdel and Mako-Mori tests also likely to exhibit more gender diversity among their writers, directors, and producers?

**Influence of Female Characters:** Does the presence of leading or strong supporting female characters correlate with passing these tests?

**Independent Films vs. Major Studio Productions:** How do independent films compare with major studio productions in terms of passing these tests?

**Public Discourse and Influence:** To what extent do films that pass these tests influence public discourse on gender representation?

**Challenging Gender Stereotypes:** Are movies that pass these tests more inclined to feature women in non-traditional roles or challenge gender stereotypes?

**Impact of Film Ratings:** How does a film’s age rating affect its likelihood of passing these tests?

**Role of Film Festivals and Awards:** What is the role of film festivals and awards in promoting films that pass these tests?

**Audience Perceptions and Demands:** How have audience perceptions and demands influenced the number of films passing these tests over time?

**Predictive Analysis:** Can data from these tests predict future trends in women's cinematic representation?

**Real-world Impact:** What real-world impact do films that pass these tests have on issues of gender equality and female empowerment?


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

## Methodology 💡
Our analysis will employ a multifaceted approach:

**Statistical Analysis:** We'll conduct thorough statistical analysis to uncover trends and correlations.

**Natural Language Processing:** NLP techniques will be applied to film summaries to detect and analyze gender stereotypes.

**Comparative Analysis:** We'll compare genres, cultures, and production types (independent vs. major studios) in terms of gender representation.

**Temporal Analysis:** Examining how trends have evolved over time, we'll use this data to potentially forecast future trends.

**Impact Assessment:** We'll assess the real-world impact of films that pass the Bechdel and Mako-Mori tests, particularly in terms of influencing public discourse and contributing to gender equality movements.

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


| Name          | Email                                 | Task                    |
|-----------------|---------------------------------------|----------------------------|
| F. Dumoncel     | `francois.dumoncel-kessler@epfl.ch`   | Preprocessing and README  |
| K. Tetard       | `kenji.tetard@epfl.ch`                | -   |
| L. Vogel        | `lena.vogel@epfl.ch`                  | -   |
| N. Dillenbourg  | `nael.dillenbourg@epfl.ch`            | -   |
| A. Bacuet       | `aymeric.bacuet@epfl.ch`              | Exploratory and data Analysis   |
