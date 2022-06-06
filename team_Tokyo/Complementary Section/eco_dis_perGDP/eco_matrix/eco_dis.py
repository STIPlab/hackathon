#!/bin/env python
# -*- encoding: utf-8 -*-
#-------------------------------------------------------------------------------
# Purpose:     economic distance matrix
# Author:      zqh
# Created:     2022-06-04
#-------------------------------------------------------------------------------

import pandas as pd
import math

data = pd.read_excel('/Users/zqh/Documents/OECD/eco_distance/perRD.xlsx', sheet_name=0)
loc_a = list(data.loc[:, "country"])
loc_data = []
for loc_1 in loc_a:
    weight_list = []
    for loc_2 in loc_a:
        value_1 = data[data['country'] == loc_1]['average'].values[0]
        value_2 = data[data['country'] == loc_2]['average'].values[0]
        weight = 1/abs(value_1-value_2)
        if math.isinf(weight):
            weight = 0
        else:
            pass
        weight_list.append(weight)
        print(loc_1+'-'+loc_2+' completed')
    loc_data.append(weight_list)

eco_weight = pd.DataFrame(loc_data, index=loc_a, columns=loc_a)
eco_weight.to_csv('/Users/zqh/Documents/OECD/eco_distance/perRD_weight.csv', encoding='gbk')
