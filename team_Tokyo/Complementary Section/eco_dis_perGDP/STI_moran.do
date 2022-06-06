/*
purpose: Moran-Index for perGDP (economic matrix) & Frequency of STIPs by country
data: 2022-06-03
*/


* Import data. Use the python script to compute matrix.
import excel using Freq_country.xlsx, firstrow clear

* Standarized matrix is for computing local Moran for each country
spatwmat using eco_dis.dta, n(W) standardize

* This one is for global Moran
spatwmat using C:\Users\ASUS\Desktop\0-1.dta, n(W_all)

* Moran-Index
spatgsa Freq, w(W_all) m two // Global
est sto Moran_global
spatlsa Freq, w(W) m two gr(moran) sy(id) id(province) // Local
est sto Moran_local

* Display the table
estat Moran_global Moran_local, b mtitle
