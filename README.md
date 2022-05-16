<div id="header" align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/OECD_logo.svg/1280px-OECD_logo.svg.png" width="350"/>
</div>

## Welcome to the OECD _Data science for STI policy_ hackathon GitHub repository!

This GitHub page provides the most important information on [how to use Github for the hackathon](#How-to-use-Github-for-the-hackathon) and [how to access the data](#How-to-access-the-data).

In case you have any questions during the hackathon, please don't hesitate to reach out to us.

## How to use Github for the hackathon

Once you have finalised your project, we would like to ask you to upload all final outputs (code, visualisations, complementary materials, etc.) to your team folder in this repository. Please do so by...

If you are not familiar with GitHub, here is a short tutorial that explains the most important functions.

[![IMAGE ALT TEXT](http://img.youtube.com/vi/USjZcfj8yxE/0.jpg)](http://www.youtube.com/watch?v=USjZcfj8yxE "Video Title")

## How to access the data



### STI strategies database

The _TIP STI strategies database_ consists of a text corpus including more than 300 STI policy strategy documents (several million words overall) from across 24 OECD countries that covers the past several years, including both the duration of the COVID-19 pandemic and the period immediately prior. The documents have been collected in collaboration with national government officials working on STI policies in a range of public administrations and have been pre-processed and machine-translated to English by the OECD.

Please follow this [link](link) to download the data in .csv-format. The dataset includes the following columns:
- a
- b

### STIP Compass policy database

The STIP Compass policy database includes qualitative and quantitative data on national STI policies. It is made up of close to 7000 initiatives from 57 countries and addresses all areas of STI policy, including initiatives spread across different ministries and national agencies, with competence over domains as broad as research, innovation, education, industry, environment, labour, finance/budget, among others. Its data is collected from a survey addressed to national government officials working on STI policies in a range of public administrations.

More detailed information about the database can be found [here](https://stip.oecd.org/stip/pages/stipDataLab). On this page you will also find the links to the STIP dataset as well as relevant descriptions and codebooks.

To easily load the data in R you can use the following code.

```bash
url <- 'https://stip.oecd.org/assets/downloads/STIP_Survey.csv'

#download the dataset
download.file(url, destfile = 'stip.csv', mode = 'wb')

#load the dataset into our working environment
stip <- read_delim('stip.csv', '|', escape_double = FALSE, trim_ws = TRUE)
```

## Useful resources

Here is a collection of links to futher useful resources:
 - https://stiplab.github.io/datastories/nlp%20tutorial/Getting%20Started%20with%20NLP%20of%20Research%20and%20Innovation%20Policy%20Data%20using%20R.html
