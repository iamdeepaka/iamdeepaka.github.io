---
layout: post
title: Bike Dataset EDA!
category: Food
---


## Importing libraries


```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score

import subprocess
```

## Data Download



```python
#code to download data
subprocess.run(["wget", "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"])
subprocess.run(["unzip","./Bike-Sharing-Dataset.zip"])
```




    CompletedProcess(args=['unzip', './Bike-Sharing-Dataset.zip'], returncode=1)



## Importing data



```python
hour_df = pd.read_csv("hour.csv")
hour_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17379 entries, 0 to 17378
    Data columns (total 17 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   instant     17379 non-null  int64  
     1   dteday      17379 non-null  object 
     2   season      17379 non-null  int64  
     3   yr          17379 non-null  int64  
     4   mnth        17379 non-null  int64  
     5   hr          17379 non-null  int64  
     6   holiday     17379 non-null  int64  
     7   weekday     17379 non-null  int64  
     8   workingday  17379 non-null  int64  
     9   weathersit  17379 non-null  int64  
     10  temp        17379 non-null  float64
     11  atemp       17379 non-null  float64
     12  hum         17379 non-null  float64
     13  windspeed   17379 non-null  float64
     14  casual      17379 non-null  int64  
     15  registered  17379 non-null  int64  
     16  cnt         17379 non-null  int64  
    dtypes: float64(4), int64(12), object(1)
    memory usage: 2.3+ MB



```python
hour_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>instant</th>
      <th>dteday</th>
      <th>season</th>
      <th>yr</th>
      <th>mnth</th>
      <th>hr</th>
      <th>holiday</th>
      <th>weekday</th>
      <th>workingday</th>
      <th>weathersit</th>
      <th>temp</th>
      <th>atemp</th>
      <th>hum</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.22</td>
      <td>0.2727</td>
      <td>0.80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2011-01-01</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0.24</td>
      <td>0.2879</td>
      <td>0.75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Data Dictionary



```python
# - instant: record index
# - dteday : date
# - season : season (1:winter, 2:spring, 3:summer, 4:fall)
# - yr : year (0: 2011, 1:2012)
# - mnth : month ( 1 to 12)
# - hr : hour (0 to 23)
# - holiday : weather day is holiday or not (extracted from [Web Link])
# - weekday : day of the week
# - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
# + weathersit :
# - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# - hum: Normalized humidity. The values are divided to 100 (max)
# - windspeed: Normalized wind speed. The values are divided to 67 (max)
# - casual: count of casual users
# - registered: count of registered users
# - cnt: count of total rental bikes including both casual and registered
```

## Preprocessing


```python
# Renaming columns names to more readable names
hour_df.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'holiday':'is_holiday',
                        'workingday':'is_workingday',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'hr':'hour',
                        'yr':'year'},inplace=True)

###########################
# Setting proper data types
###########################
# date time conversion
hour_df['datetime'] = pd.to_datetime(hour_df.datetime)

# categorical variables
hour_df['season'] = hour_df.season.astype('category')
hour_df['is_holiday'] = hour_df.is_holiday.astype('category')
hour_df['weekday'] = hour_df.weekday.astype('category')
hour_df['weather_condition'] = hour_df.weather_condition.astype('category')
hour_df['is_workingday'] = hour_df.is_workingday.astype('category')
hour_df['month'] = hour_df.month.astype('category')
hour_df['year'] = hour_df.year.astype('category')
hour_df['hour'] = hour_df.hour.astype('category')
```

## Plotting


```python
# Configuring plotting visual and sizes
sns.set_style('whitegrid')
sns.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)
```


```python
plt.figure(figsize=(15,10))
plt.tight_layout()
sns.distplot(hour_df['total_count']);
```


![png](output_13_0.png)



```python
# map day of week to strings for better understanding
season_mapping = {1:"winter", 2:"spring", 3:"summer", 4:"fall"}
hour_df['season_name'] = hour_df['season'].map(season_mapping)
```


```python
fig,ax = plt.subplots()
sns.pointplot(data=hour_df[['hour',
                           'total_count',
                           'season_name']],
              x='hour',
              y='total_count',
              hue='season_name',
              ax=ax)
ax.set(title="Season wise hourly distribution of counts")
```




    [Text(0.5, 1.0, 'Season wise hourly distribution of counts')]




![png](output_15_1.png)



```python
fig,ax = plt.subplots()
sns.barplot(data=hour_df[['month',
                           'total_count']],
              x='month',
              y='total_count',
              ax=ax)
ax.set(title="Monthly distribution of counts")
```




    [Text(0.5, 1.0, 'Monthly distribution of counts')]




![png](output_16_1.png)



```python
fig,ax = plt.subplots()
sns.barplot(data=hour_df[['season_name',
                           'total_count']],
              x='season_name',
              y='total_count',
              ax=ax)
ax.set(title="Seasonal distribution of counts")
```




    [Text(0.5, 1.0, 'Seasonal distribution of counts')]




![png](output_17_1.png)


#### Checking for outliers:


```python
fig,(ax1,ax2) = plt.subplots(ncols=2)
sns.boxplot(data=hour_df[['total_count',
                          'casual',
                          'registered']],ax=ax1)
sns.boxplot(data=hour_df[['temp',
                          'windspeed']],ax=ax2)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f4487f92a90>




![png](output_19_1.png)



```python
fig,ax = plt.subplots()
sns.boxplot(data=hour_df[['total_count',
                          'hour']],x='hour',y='total_count',ax=ax)
ax.set(title="Checking for outliners in day hours")
```




    [Text(0.5, 1.0, 'Checking for outliners in day hours')]




![png](output_20_1.png)


#### Correlations


```python
corrMatt = hour_df[['temp',
                    'humidity', 
                    'windspeed',  
                    'total_count']].corr()

mask = np.array(corrMatt)
# Turning the lower-triangle of the array to false
mask[np.tril_indices_from(mask)] = False
fig,ax = plt.subplots()
sns.heatmap(corrMatt, 
            mask=mask,
            vmax=.8, 
            square=True,
            annot=True,
            ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f447ffa9650>




![png](output_22_1.png)



```python
plt.figure(figsize=(15,10))

# The columns of the explanatory and target variables
cols = ['temp','humidity', 'windspeed', 'total_count']

# Plot pairwise relationships between 'temp' and 'cnt'
sns.pairplot(hour_df[cols], size=5)
plt.tight_layout();
plt.show();
```


    <Figure size 1080x720 with 0 Axes>



![png](output_23_1.png)


## Feature Engineering
#### **Since the dataset contains multiple categorical variables, it is imperative that we encode the nominal ones before we use them in our modeling process.**


```python
# Defining categorical variables encoder method
def fit_transform_ohe(df,col_name):
    """This function performs one hot encoding for the specified column.
    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded
    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series
    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name+'_label'] = le_labels
    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    return le,ohe,features_df

# given label encoder and one hot encoder objects, 
# encode attribute to ohe
def transform_ohe(df,le,ohe,col_name):
    """This function performs one hot encoding for the specified
        column using the specified encoder objects.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the onen hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded

    Returns:
        tuple: transformed column as pandas Series

    """
    # label encode
    col_labels = le.transform(df[col_name])
    df[col_name+'_label'] = col_labels
    
    # ohe 
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return features_df
```

## Train-Test Split


```python
# Divide the dataset into training and testing sets
X, X_test, y, y_test = train_test_split(hour_df.iloc[:,0:-3],
                                        hour_df.iloc[:,-1],
                                        test_size=0.33,
                                        random_state=42)
X.reset_index(inplace=True)
y = y.reset_index()

X_test.reset_index(inplace=True)
y_test = y_test.reset_index()
```


```python
# Encoding all the categorical features
cat_attr_list = ['season','is_holiday',
                 'weather_condition','is_workingday',
                 'hour','weekday','month','year']
# though we have transformed all categoricals into their one-hot encodings, note that ordinal
# attributes such as hour, weekday, and so on do not require such encoding.
numeric_feature_cols = ['temp','humidity','windspeed',
                        'hour','weekday','month','year']
subset_cat_features =  ['season','is_holiday','weather_condition','is_workingday']

###############
# Train dataset
###############
encoded_attr_list = []
for col in cat_attr_list:
    return_obj = fit_transform_ohe(X,col)
    encoded_attr_list.append({'label_enc':return_obj[0],
                              'ohe_enc':return_obj[1],
                              'feature_df':return_obj[2],
                              'col_name':col})


feature_df_list  = [X[numeric_feature_cols]]
feature_df_list.extend([enc['feature_df'] \
                        for enc in encoded_attr_list \
                        if enc['col_name'] in subset_cat_features])

train_df_new = pd.concat(feature_df_list, axis=1)
print("Train dataset shape::{}".format(train_df_new.shape))
print(train_df_new.head())

##############
# Test dataset
##############
test_encoded_attr_list = []
for enc in encoded_attr_list:
    col_name = enc['col_name']
    le = enc['label_enc']
    ohe = enc['ohe_enc']
    test_encoded_attr_list.append({'feature_df':transform_ohe(X_test,
                                                              le,ohe,
                                                              col_name),
                                   'col_name':col_name})
    
    
test_feature_df_list = [X_test[numeric_feature_cols]]
test_feature_df_list.extend([enc['feature_df'] \
                             for enc in test_encoded_attr_list \
                             if enc['col_name'] in subset_cat_features])

test_df_new = pd.concat(test_feature_df_list, axis=1) 
print("Test dataset shape::{}".format(test_df_new.shape))
print(test_df_new.head())
```

    Train dataset shape::(11643, 19)
       temp  humidity  windspeed hour weekday month year  season_1  season_2  \
    0  0.64      0.65     0.1940    0       5     9    0       0.0       0.0   
    1  0.50      0.45     0.2239   13       2     3    0       0.0       1.0   
    2  0.86      0.47     0.5224   12       0     8    1       0.0       0.0   
    3  0.30      0.61     0.0000    2       3     2    1       1.0       0.0   
    4  0.54      0.19     0.4179   17       6     4    1       0.0       1.0   
    
       season_3  season_4  is_holiday_0  is_holiday_1  weather_condition_1  \
    0       1.0       0.0           1.0           0.0                  1.0   
    1       0.0       0.0           1.0           0.0                  1.0   
    2       1.0       0.0           1.0           0.0                  1.0   
    3       0.0       0.0           1.0           0.0                  1.0   
    4       0.0       0.0           1.0           0.0                  1.0   
    
       weather_condition_2  weather_condition_3  weather_condition_4  \
    0                  0.0                  0.0                  0.0   
    1                  0.0                  0.0                  0.0   
    2                  0.0                  0.0                  0.0   
    3                  0.0                  0.0                  0.0   
    4                  0.0                  0.0                  0.0   
    
       is_workingday_0  is_workingday_1  
    0              0.0              1.0  
    1              0.0              1.0  
    2              1.0              0.0  
    3              0.0              1.0  
    4              1.0              0.0  
    Test dataset shape::(5736, 19)
       temp  humidity  windspeed hour weekday month year  season_1  season_2  \
    0  0.80      0.27     0.1940   19       6     6    1       0.0       0.0   
    1  0.24      0.41     0.2239   20       1     1    1       1.0       0.0   
    2  0.32      0.66     0.2836    2       5    10    0       0.0       0.0   
    3  0.78      0.52     0.3582   19       2     5    1       0.0       1.0   
    4  0.26      0.56     0.3881    0       4     1    0       1.0       0.0   
    
       season_3  season_4  is_holiday_0  is_holiday_1  weather_condition_1  \
    0       1.0       0.0           1.0           0.0                  1.0   
    1       0.0       0.0           0.0           1.0                  1.0   
    2       0.0       1.0           1.0           0.0                  1.0   
    3       0.0       0.0           1.0           0.0                  1.0   
    4       0.0       0.0           1.0           0.0                  1.0   
    
       weather_condition_2  weather_condition_3  weather_condition_4  \
    0                  0.0                  0.0                  0.0   
    1                  0.0                  0.0                  0.0   
    2                  0.0                  0.0                  0.0   
    3                  0.0                  0.0                  0.0   
    4                  0.0                  0.0                  0.0   
    
       is_workingday_0  is_workingday_1  
    0              1.0              0.0  
    1              1.0              0.0  
    2              0.0              1.0  
    3              0.0              1.0  
    4              0.0              1.0  


## Modeling


```python
X = train_df_new
y = y.total_count.values.reshape(-1,1)

lin_reg = linear_model.LinearRegression()

# using the k-fold cross validation (specifically 10-fold) to reduce overfitting affects
# cross_val_predict function returns cross validated prediction values as fitted by the model object.
predicted = cross_val_predict(lin_reg, X, y, cv=10)
```


```python
# Analysing residuals in our predictinos
fig,ax = plt.subplots(figsize=(15,15))
ax.scatter(y, y-predicted)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
ax.set_title('Residual Plot')
plt.show()
```


![png](output_31_0.png)



```python
# Evaluating model in cross-validation iteration

r2_scores = cross_val_score(lin_reg, X, y, cv=10)
mse = cross_val_score(lin_reg, X, y, cv=10,scoring='neg_mean_squared_error')

fig,ax = plt.subplots()
ax.plot(range(0,10),
        r2_scores)
ax.set_xlabel('Iteration')
ax.set_ylabel('R.Squared')
ax.set_title('Cross-Validation scores')
plt.show()


print("R-squared::{}".format(r2_scores))
print("MSE::{}".format(mse))
```


![png](output_32_0.png)


    R-squared::[0.39894459 0.35575732 0.3873037  0.38796861 0.42489499 0.41571164
     0.37379762 0.39339864 0.39589746 0.40871611]
    MSE::[-19612.38349313 -20800.77110185 -20256.54013607 -18545.99033804
     -18746.57816436 -21015.35560028 -21549.12876053 -21567.27946203
     -21044.42416385 -18899.05989574]


## Testing dataset evaluation


```python
# Predict model based on training dataset
lin_reg.fit(X,y)

# Constructing test dataset
X_test = test_df_new
y_test = y_test.total_count.values.reshape(-1,1)


y_pred = lin_reg.predict(X_test)
residuals = y_test-y_pred

fig, ax = plt.subplots(figsize=(15,15))
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot with R-Squared={}".format(np.average(lin_reg.score(X_test,y_test))))
plt.show()

print("MSE: {}".format(metrics.mean_squared_error(y_test, y_pred)))
```


![png](output_34_0.png)


    MSE: 19062.99975600927



```python
print('Coefficients of linear regression: \n')
dict(zip(X_test.columns.tolist(), lin_reg.coef_[0]))
```

    Coefficientsof linear regression: 
    





    {'temp': 347.2038429288963,
     'humidity': -194.41029021137572,
     'windspeed': 24.754603714188335,
     'hour': 7.410754277779705,
     'weekday': 1.537774939386947,
     'month': -0.07003609078512056,
     'year': 81.97535032363254,
     'season_1': -21.883515856532508,
     'season_2': 0.406942786740506,
     'season_3': -24.538641381444965,
     'season_4': 46.015214451237064,
     'is_holiday_0': 14.194260232269672,
     'is_holiday_1': -14.194260232269704,
     'weather_condition_1': -6.017713451173841,
     'weather_condition_2': 2.8648273000590763,
     'weather_condition_3': -32.15613007910385,
     'weather_condition_4': 35.30901623021856,
     'is_workingday_0': -1.6922509610512657,
     'is_workingday_1': 1.6922509610512106}




```python
# The mean squared error
print('Mean squared error: %.2f'
      % metrics.mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % metrics.r2_score(y_test, y_pred))
```

    Mean squared error: 19063.00
    Coefficient of determination: 0.40


## **As we can cleary see, the performance is dismal due to non-linearity of independent variables to dependent features and we could only predict around 40 percent of the outcomes.**


```python

```
