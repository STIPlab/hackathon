<div id="header" align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/OECD_logo.svg/1280px-OECD_logo.svg.png" width="350"/>
</div>

# Welcome to the OECD _Data science for STI policy_ hackathon GitHub repository!

This GitHub page provides the most important information on [how to use Github for the hackathon](#How-to-use-Github-for-the-hackathon) and [how to access the data](#How-to-access-the-data). Please make sure to upload your final project outputs (code, visualisations, complementary materials, etc.) to your team folder in this repository.

In case you have any questions during the hackathon, please don't hesitate to reach out to us.

# How to use Github for the hackathon

[![IMAGE ALT TEXT](http://img.youtube.com/vi/USjZcfj8yxE/0.jpg)](http://www.youtube.com/watch?v=USjZcfj8yxE "Video Title")

# How to access the data



## STI strategies database

The TIP STI strategies database consists of a text corpus including 314  STI policy strategy documents (several million words overall) from across 24 OECD countries that covers the past several years, including both the duration of the COVID-19 pandemic and the period immediately prior. The documents have been collected in collaboration with national government officials working on STI policies in a range of public administrations and have been pre-processed and machine-translated to English by the OECD. In addition, the OECD team is currently developing an annotated version of the corpus that identifies key themes in the text and might be used by participants to train supervised language models.

- a
- b

## STIP Compass policy database

The STIP Compass policy database includes qualitative and quantitative data on national STI policies. It is made up of close to 7000 initiatives from 57 countries and addresses all areas of STI policy, including initiatives spread across different ministries and national agencies, with competence over domains as broad as research, innovation, education, industry, environment, labour, finance/budget, among others. Its data is collected from a survey addressed to national government officials working on STI policies in a range of public administrations.

- https://stip.oecd.org/stip/pages/stipDataLab
- https://stip.oecd.org/stip/data-stories

```bash
url <- 'https://stip.oecd.org/assets/downloads/STIP_Survey.csv'

#download the dataset
download.file(url, destfile = 'stip.csv', mode = 'wb')

#load the dataset into our working environment
stip <- read_delim('stip.csv', '|', escape_double = FALSE, trim_ws = TRUE)
```

# Useful resources
