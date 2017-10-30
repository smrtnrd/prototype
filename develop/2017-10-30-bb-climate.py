
# coding: utf-8

# ## Data Preparation & Exploration for Climate (ulmo)
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


# In[89]:

#test
ili_test = pd.read_csv("../data/raw.csv")

x = ili_test['Latitude'].drop_duplicates()
y = ili_test['Longitude'].drop_duplicates()
state = ili_test['statename'].drop_duplicates()
ili_climate = {}

y.head()


# In[ ]:

for i in range(len(ili_test['Latitude'])):
    s=state[i]
    ili_climate[s] = daymet.get_daymet_singlepixel(longitude=y[i], latitude=x[i], 
                                   years=[2010,2015])
print(ili_climate)


# In[13]:



res = Epidata.fluview(['nat'], [201440, Epidata.range(201501, 201510)])
print(res['result'], res['message'], len(res['epidata']))


# In[121]:

#test
ornl_lat, ornl_long = 35.9313167, -84.3104124
df = daymet.get_daymet_singlepixel(longitude=ornl_long, latitude=ornl_lat, 
                                   years=[2012,2013])
df.head()


# In[122]:

df.index.year


# In[123]:

df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
#df.drop('index', axis=0, inplace=True)
df.head()


# [Pandas dataframe groupeby datetime month](https://stackoverflow.com/questions/24082784/pandas-dataframe-groupby-datetime-month)
# 

# In[124]:

df.shape


# In[127]:

# Group the data by month, and take the mean for each group (i.e. each month)
df[['prcp', 'tmax', 'tmin']].resample('M').mean().add_prefix('mean_')


# In[110]:

df_month = df[['month','year', 'yday', 'prcp', 'tmax', 'tmin']].groupby(['month', 'year']).mean()
df_month.head()


# In[105]:

df_month.shape


# We have :
# -  categorical variable : year, yday(integer)
# -  other variable : floats or interger
# -  index : datetime

# In[12]:

df.info()


# In[75]:

#load ILI data from csv
ili = pd.read_csv("../data/raw.csv")
ili.head()


# In[76]:

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


# In[70]:

ili.head()


# In[77]:

ili.shape


# In[78]:

ili.drop_duplicates()
ili.shape


# In[79]:

ili.info()


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



