<div id="header" align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/OECD_logo.svg/1280px-OECD_logo.svg.png" width="350"/>
</div>

## Welcome to the OECD _Data science for STI policy_ hackathon GitHub repository!

This GitHub page provides the most important information on [how to use Github for the hackathon](#How-to-use-Github-for-the-hackathon) and [how to access the data](#How-to-access-the-data).

In case you have any questions during the hackathon, please don't hesitate to reach out to us: Jan.EINHOFF@oecd.org; Andres.BARRENECHE@oecd.org; Maria.FLEMING@oecd.org; Blandine.SERVE@oecd.org. 

## How to use Github for the hackathon

Once you have finalised your project, we would like to ask you to upload all final outputs (code, visualisations, complementary materials, etc.) to your team folder in this repository. Please do so by dragging all files to the respective team folder or by using the GitHub desktop environment. To be able to upload the data, please create a GitHub account and let us know your credetials so we can add you as a collaborator. Please note that GitHub will not accept individual files that are larger than 50MB.

*Add Issues for Q&A*

## How to access the data

Please find below a short description of the two data sources as well as instructions on how to access the data.

### STI strategies database

The _TIP STI strategies database_ consists of a text corpus including more than 300 STI policy strategy documents (several million words overall) from across 24 OECD countries that covers the past several years, including both the duration of the COVID-19 pandemic and the period immediately prior. The documents have been collected in collaboration with national government officials working on STI policies in a range of public administrations and have been pre-processed and machine-translated to English by the OECD.

The dataset includes the following columns:
* _country_: Name of the country that issued the document
* _period_: Indicator for whether the document was issued before or during the COVID-19 pandemic
* _year_: Year when the document was issued
* _doc_id_: Identifier of the document
* _title_: Title of the document
* _language_: Original lanugage of the document
* _text raw_: Translated text of the document
* _text clean_: Translated and cleaned text of the document (no numbers and punctuation, no stopwords, lemmatization, n-grams)

Please reach out if you require the original, untranslated documents for your analyses.

You can download the data in .csv-format from the resources folder in this repository. Alternatively, you can use the following Python code to load the data into a Pandas dataframe:

```bash
import pandas as pd

#download the dataset
url = "https://raw.githubusercontent.com/STIPlab/hackathon/main/resources/STI%20strategies_database.csv"
data_github = pd.read_csv(url)
```

To load the data in R you can also use the following code:

```bash
library(RCurl)

#download the dataset
url <- getURL("https://raw.githubusercontent.com/STIPlab/hackathon/main/resources/STI%20strategies_database.csv")
data_github <- read.csv(text = url)
```

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
