<div id="header" align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/OECD_logo.svg/1280px-OECD_logo.svg.png" width="300"/>
</div>

# Welcome to the OECD hackathon: _Data science for STI policy_!

This GitHub page provides the most important information on [How to use Github for the hackathon](#How-to-use-Github-for-the-hackathon) and [How to access the data](#How-to-access-the-data). Please make sure to upload your final project outputs (code, visualisations, complementary materials, etc.) to one of the folders in this repository.

# How to use Github for the hackathon

# How to access the data

## STI policy strategies

- a
- b

## STI policy initiatives

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
