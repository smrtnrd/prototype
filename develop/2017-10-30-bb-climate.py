
# coding: utf-8

# # Data Preparation & Exploration for Climate (ulmo)
# 
# I want to merge the data from GSOD to my ILI data by getting a summary of the data by position
# -  exploring daymet : daily temperature, precipitation for any locationsion in the US [reference](https://github.com/ulmo-dev/ulmo/blob/master/examples/Using%20Daymet%20weather%20data%20from%20ORNL%20webservice.ipynb)

# In[1]:

# import packages and modules
from ulmo.nasa import daymet
from delphi_epidata import Epidata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import shuffle

pd.set_option('display.max_columns', 100)


# In[7]:

#test
ili_test = pd.read_csv("../data/raw.csv")
ili_test.head()


# In[39]:

coordinate = ili_test[['statename','Latitude','Longitude']]
coordinate = coordinate.drop_duplicates()
coordinate = coordinate.reset_index(drop=True)
coordinate.head()
#print("the data contains {} rows".format(len(coordinate)))


# In[123]:

ili_test.shape


# -  The latitude (WGS84), value between 52.0 and 14.5.
# -  The longitude (WGS84), value between -131.0 and -53.0.

# In[43]:

coordinate = coordinate[coordinate.Longitude >= -131.0]
coordinate = coordinate[coordinate.Longitude <= -53.0]
coordinate = coordinate[coordinate.Latitude >= 14.5]
coordinate = coordinate[coordinate.Latitude <= 52.0]
coordinate = coordinate.reset_index(drop=True)
coordinate


# In[89]:

climate = []
for i in range(len(coordinate.statename)):
    df = daymet.get_daymet_singlepixel(longitude=coordinate.Longitude[i], latitude=coordinate.Latitude[i], 
                                   years=[2010,2015])
    df['statename'] = coordinate.statename[i]
    df['Latitude'] = coordinate.Latitude[i]
    df['Longitude'] = coordinate.Longitude[i]
    climate.append(df)
    
climate = pd.concat(climate)

climate['year'] = climate.index.year
climate['month'] = climate.index.month
climate['day'] = climate.index.day

climate.head()


# In[46]:

# save the file
climate.to_csv("../data/climate.csv", sep='\t', encoding='utf-8')


# In[47]:

res = Epidata.fluview(['nat'], [201440, Epidata.range(201501, 201510)])
print(res['result'], res['message'], len(res['epidata']))


# In[48]:

#test
#ornl_lat, ornl_long = 35.9313167, -84.3104124
#df = daymet.get_daymet_singlepixel(longitude=ornl_long, latitude=ornl_lat, 
#                                   years=[2012,2013])


# In[49]:

#df.index.year


# In[90]:

climate['year'] = climate.index.year
climate['month'] = climate.index.month
climate['day'] = climate.index.day
#df.drop('index', axis=0, inplace=True)
climate.head()


# [Pandas dataframe groupeby datetime month](https://stackoverflow.com/questions/24082784/pandas-dataframe-groupby-datetime-month)
# 

# In[91]:

climate.shape


# In[92]:

df_month = climate[['month','year', 'yday', 'prcp', 'tmax', 'tmin','Latitude','Longitude','statename']].groupby(['statename','year', 'month',], as_index = False).mean()
df_month = df_month.rename(columns={'yday':'day',
                                    'prcp':'mean_prcp',
                                   'tmax': 'mean_tmax',
                                   'tmin':'mean_tmin'})
df_month.head()


# In[93]:

df_month.shape


# We have :
# -  categorical variable : year, yday(integer)
# -  other variable : floats or interger
# -  index : datetime

# In[135]:

df.info()


# In[95]:

#load ILI data from csv
ili = pd.read_csv("../data/raw.csv")
ili.head()


# In[96]:

# Clean the data 
# tramsform to datetime
ili['weekend'] = pd.to_datetime(ili['weekend'], format='%b-%d-%Y')
ili['season'] = pd.to_datetime(ili['season'], format='%Y-%y') 

# add year month and day in the data
ili['year'] = ili.weekend.dt.year
ili['month'] = ili.weekend.dt.month
ili['day'] = ili.weekend.dt.day

#remove data that we don't need
ili.drop(['weekend','season','weeknumber'], axis=1, inplace=True)


# In[97]:

ili.head()


# In[98]:

ili.shape


# In[99]:

ili.drop_duplicates()
ili.shape


#  ## Merge data

# In[126]:

df = pd.merge(ili, df_month, on = ['statename', 'year', 'month', 'Latitude', 'Longitude'  ], how = 'left')
df


# In[125]:

df.shape


# ## Metadata
# 
# store meta-information about the variables in a DataFrame
# 
# 
# -  **role**: response, explanatory (variable that we want to predict)
# -  **level**: nominal, interval, ordinal, binary
# -  **keep**: True or False
# -  **dtype**: int, float, str

# In[56]:

meta_ili = []
for f in ili.columns:
    # Defining the role
    if f == 'activity_level' or f == 'activity_level_label':
        role = 'response'
    else:
        role = 'explanatory'
         
    # Defining the level
    if 'statename' == f :
        level = 'nominal'
    elif 'activity_level_label' == f:
        level = 'ordinal'
    elif 'weekend' == f or 'season' == f or 'weeknumber' == f:
        level = 'interval'
    elif ili[f].dtype == float:
        level = 'ordinal'
    elif ili[f].dtype == int:
        level = 'ordinal'
        
    # Initialize keep to True for all variables except for id
    keep = True
    if f == 'activity_level_label':
        keep = False
    
    # Defining the data type 
    dtype = ili[f].dtype
    
    # Creating a Dict that contains all the metadata for the variable
    f_dict = {
        'varname': f,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    meta_ili.append(f_dict)
    
meta = pd.DataFrame(meta_ili, columns=['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace=True)


# In[57]:

meta           


# In[58]:

# Below the number of variables per role and level are displayed.

pd.DataFrame({'count' : meta.groupby(['role', 'level'])['role'].size()}).reset_index()


# ### interval variables

# In[59]:

v = meta[(meta.level == 'ordinal') & (meta.keep)].index
ili[v].describe()


# In[60]:

v = meta[(meta.level == 'nominal') & (meta.keep)].index
ili[v].describe()


# ### Checking the cardinality of the categorical variables

# In[62]:

v = meta[(meta.level == 'interval') & (meta.keep)].index

for f in v:
    dist_values = ili[f].value_counts().shape[0]
    print('Variable {} has {} distinct values'.format(f, dist_values))


# In[ ]:




# In[ ]:



