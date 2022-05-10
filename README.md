<div id="header" align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/OECD_logo.svg/1280px-OECD_logo.svg.png" width="300"/>
</div>

# Welcome

# Timeline

- [Timeline](#Timeline)  
  - [Datasets](#Datasets)  
  - [Other useful resources](#Other)   
<a name="headers"/>


# How to use Github for the hackathon

# Datasets

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
