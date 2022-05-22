<div id="header" align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/OECD_logo.svg/1280px-OECD_logo.svg.png" width="350"/>
</div>

## Welcome to the OECD _Data science for STI policy_ hackathon GitHub repository!

This GitHub page provides the most important information on [how to use Github for the hackathon](#How-to-use-Github-for-the-hackathon) and [how to access the data](#How-to-access-the-data).

In case you have any **questions during the hackathon**, please use this repository's [issues tab](https://github.com/STIPlab/hackathon/issues) and tag your post with one of the existing labels so the right persons are notified.

## How to use Github for the hackathon

Once you have finalised your project, we would like to ask you to upload all final outputs (code, visualisations, complementary materials, etc.) to your team folder in this repository. Please do so by dragging all files to the respective team folder or by using the GitHub desktop environment. To be able to upload the data, please create a GitHub account and let us know your credentials so we can add you as a collaborator. Please note that GitHub will not accept individual files that are larger than 50MB.

During the hackathon we will use this GitHub respository to answer any questions you might have and to share the answers with the other participants. Please navigate to the 'Issues' tab on the main page of the repository and leave your question by creating a new issue. We will try to answer each question as soon as possible.

## How to access the data

Please find below a short description of the two data sources as well as instructions on how to access the data.

### STI strategies database

The _TIP STI strategies database_ consists of a text corpus including more than 300 STI policy strategy documents (several million words overall) from across 24 OECD countries that covers the past several years, including both the duration of the COVID-19 pandemic and the period immediately prior. The documents have been collected in collaboration with national government officials working on STI policies in a range of public administrations and have been pre-processed and machine-translated to English by the OECD.

The dataset includes the following columns:
* _country_: Name of the country that issued the document
* _year_: Year when the document was issued
* _period_: Indicator for whether the document was issued before or during the COVID-19 pandemic
* _doc_id_: Identifier of the document
* _title_: Title of the document
* _text original_: Original text of the document
* _text translated_: Translated text of the document
* _text clean_: Translated and cleaned text of the document (no numbers and punctuation, no stopwords, lemmatization, n-grams)

You can download the data in .RData-format [here](https://www.dropbox.com/s/vd4ky6kv1a3cmho/strategies_final.RData?dl=0). The dataset is quite large which is why we use the .RData-format. You can easily open the file in R by using the load()-command or by using the pyreadr package in Python.

### STIP Compass policy database

The STIP Compass policy database includes qualitative data on national STI policies. It is made up of close to 7000 initiatives from 57 countries and the European Union. The database covers all areas of STI policy, including initiatives spread across different ministries and national agencies, with competence over domains as broad as research, innovation, education, industry, environment, labour, finance/budget, among others. Its data is collected from a survey addressed to national government officials working on STI policies in a range of public administrations.

A few essential details about the dataset:

* The data model used to structure STIP Compass can understood by viewing this [PDF file](https://stip.oecd.org/assets/downloads/STIPCompassTaxonomies.pdf) and the accompanying [codebook](https://stiplab.github.io/hackathon/resources/2021%20STIP%20survey%20codebook.xlsx). 
* The dataset has two header rows. The first row contains the variable names, whereas the second row includes a short description of the variable.
* After the headers, each row provides data for a given initiative and instrument. As an initiative can have more than one instrument, subsequent rows can contain information on multiple instruments from the same initiative.
* If you plan to load a CSV file, please select UTF-8 encoding and indicate the pipe character '|' (without quotes) as separator.

More detailed information about the database can be found [here](https://stip.oecd.org/stip/pages/stipDataLab).

To load the data in Python into a Pandas dataframe you can use the following code:

```bash
import pandas as pd

#download the dataset
url = 'https://stip.oecd.org/assets/downloads/STIP_Survey.csv'
compass_df = pd.read_csv(url, sep='|', encoding='UTF-8-SIG', header=0, low_memory=False)
```

You can also easily load the data in R using the following code:

```bash
library(readr)

url <- 'https://stip.oecd.org/assets/downloads/STIP_Survey.csv'

#download the dataset
download.file(url, destfile = 'stip.csv', mode = 'wb')

#load the dataset into our working environment
stip <- read_delim('stip.csv', '|', escape_double = FALSE, trim_ws = TRUE)
```

You may inspect and re-use code found in these projects:

* [Natural Language Processing of Research and Innovation Policy Data using R](https://stiplab.github.io/datastories/nlp%20tutorial/Getting%20Started%20with%20NLP%20of%20Research%20and%20Innovation%20Policy%20Data%20using%20R.html)
* [Pilot analysis in Python comparing country policies in support of public research, **do not circulate**](https://stiplab.github.io/datastories/comparing%20countries/Comparing%20country%20policies%20using%20STIP%20Compass%20%5Bdraft%5D.html)

### STI.Scoreboard

A possible source of complementary data is the [STI.Scoreboard](https://www.oecd.org/sti/scoreboard.htm) infrastructure. It contains over 1000 indicators on research and development, science, business innovation, patents, education and the economy, drawing on the very latest, quality assured statistics from OECD and partner international organisations. These indicators are accessible via a dedicated API that uses SDMX queries. The following Python and R tutorials provide more information and include the necessary code to access this infrastructure.

#### How to retrieve STI.Scoreboard indicators found in STIP Compass using Python and SDMX?
* [Python tutorial](https://colab.research.google.com/drive/1-qPUqDh6QolHOauYf1pt4GWMW8M_yvzk?usp=sharing)
* [R tutorial](https://colab.research.google.com/drive/168-JCLtK8PCduEmRY4x22qEVW1F0q0vc?usp=sharing)
