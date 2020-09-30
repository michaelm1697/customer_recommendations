```python
## Importing needed libraries
import pandas as pd
import datetime
import numpy as np
import random
%matplotlib inline
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
```


```python
## Importing needed files
directory = 'C:/Users/viks9/Downloads/retailrocket/'

def importData(file):
    return pd.read_csv(directory + file + '.csv')

category_tree = importData('category_tree')
events = importData('events')
items1 = importData('item_properties_part1')
items2 = importData('item_properties_part2')
items_properties = pd.concat([items1, items2])
```

### Exploring and Cleaning Data


```python
category_tree.head()
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
      <th>categoryid</th>
      <th>parentid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1016</td>
      <td>213.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>809</td>
      <td>169.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>570</td>
      <td>9.0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1691</td>
      <td>885.0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>536</td>
      <td>1691.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
events.head()
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
      <th>timestamp</th>
      <th>visitorid</th>
      <th>event</th>
      <th>itemid</th>
      <th>transactionid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1433221332117</td>
      <td>257597</td>
      <td>view</td>
      <td>355908</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1433224214164</td>
      <td>992329</td>
      <td>view</td>
      <td>248676</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1433221999827</td>
      <td>111016</td>
      <td>view</td>
      <td>318965</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1433221955914</td>
      <td>483717</td>
      <td>view</td>
      <td>253185</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1433221337106</td>
      <td>951259</td>
      <td>view</td>
      <td>367447</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Converting time stamp columns from UNIX format and sorting values by the timestamp
import datetime
events_times =[]
for i in events['timestamp']:
    events_times.append(datetime.datetime.fromtimestamp(i//1000.0)) 
events['timestamp'] = events_times

events = events.sort_values(by = 'timestamp')
```


```python
events.isnull().sum()
```




    timestamp              0
    visitorid              0
    event                  0
    itemid                 0
    transactionid    2733644
    dtype: int64



We have 2733644 null values in the transactionid column, we will explore this more below.


```python
events[events['transactionid'].isnull()].head()
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
      <th>timestamp</th>
      <th>visitorid</th>
      <th>event</th>
      <th>itemid</th>
      <th>transactionid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1462974</td>
      <td>2015-05-02 22:00:04</td>
      <td>693516</td>
      <td>addtocart</td>
      <td>297662</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1464806</td>
      <td>2015-05-02 22:00:11</td>
      <td>829044</td>
      <td>view</td>
      <td>60987</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1463000</td>
      <td>2015-05-02 22:00:13</td>
      <td>652699</td>
      <td>view</td>
      <td>252860</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1465287</td>
      <td>2015-05-02 22:00:24</td>
      <td>1125936</td>
      <td>view</td>
      <td>33661</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1462955</td>
      <td>2015-05-02 22:00:26</td>
      <td>693516</td>
      <td>view</td>
      <td>297662</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
events['event'].value_counts()
```




    view           2664312
    addtocart        69332
    transaction      22457
    Name: event, dtype: int64




```python
## Calculating amount of transactions
non_transaction_events = events['event'].size - events['event'].value_counts()['transaction']
print("There were " + str(non_transaction_events) + " events that did not result in a transaction.")
```

    There were 2733644 events that did not result in a transaction.
    

Null transaction ID's occur when an event is a view or addedtocart.


```python
## Calculating percentage of events that resulted in a transaction
percent_purchased = len(events[events['event'] == 'transaction']) / len(events[events['event'] != 'transaction']) * 100
print("Only " + str("{:.2f}".format(percent_purchased)) + "% of events resulted in a purchase")
```

    Only 0.82% of events resulted in a purchase
    


```python
items_properties.head()
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
      <th>timestamp</th>
      <th>itemid</th>
      <th>property</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1435460400000</td>
      <td>460429</td>
      <td>categoryid</td>
      <td>1338</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1441508400000</td>
      <td>206783</td>
      <td>888</td>
      <td>1116713 960601 n277.200</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1439089200000</td>
      <td>395014</td>
      <td>400</td>
      <td>n552.000 639502 n720.000 424566</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1431226800000</td>
      <td>59481</td>
      <td>790</td>
      <td>n15360.000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1431831600000</td>
      <td>156781</td>
      <td>917</td>
      <td>828513</td>
    </tr>
  </tbody>
</table>
</div>



The category tree dataframe does not have an itemid for us to reference, so we will get the categoryid's from the items_properties dataframe. When the property is categoryid, the value will be the items categoryid. 


```python
## Creating a dataframe for category ids
category_ids = items_properties[items_properties['property'] == 'categoryid']
category_ids = category_ids.rename(columns={"value": "categoryid"})
category_ids = category_ids.drop(['timestamp', 'property'], axis = 1)
category_ids = category_ids.set_index('itemid')
category_ids.head()
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
      <th>categoryid</th>
    </tr>
    <tr>
      <th>itemid</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>460429</td>
      <td>1338</td>
    </tr>
    <tr>
      <td>281245</td>
      <td>1277</td>
    </tr>
    <tr>
      <td>35575</td>
      <td>1059</td>
    </tr>
    <tr>
      <td>8313</td>
      <td>1147</td>
    </tr>
    <tr>
      <td>55102</td>
      <td>47</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Converting time stamp columns from UNIX format and sorting values by the timestamp
items_times = []
for i in items_properties['timestamp']:
    items_times.append(datetime.datetime.fromtimestamp(i//1000.0)) 
items_properties['timestamp'] = items_times
```

## Splitting Events Dataframe for events that resulted in transaction and did not result in a transaction

### Resulted in a transaction



```python
## Customers that made a transaction
customer_purchased = (events['visitorid'][events['event'] == 'transaction'].unique())

print(str(len(customer_purchased)) + (" customers made ") + str((events['event'] == 'transaction').sum()) + " purchases")
```

    11719 customers made 22457 purchases
    


```python
## Creating a dataframe of the events for visitors that made a purchase
transaction_df = events[events['visitorid'].isin(customer_purchased)]
transaction_df.head()
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
      <th>timestamp</th>
      <th>visitorid</th>
      <th>event</th>
      <th>itemid</th>
      <th>transactionid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1462684</td>
      <td>2015-05-02 22:03:20</td>
      <td>41386</td>
      <td>view</td>
      <td>340921</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1462673</td>
      <td>2015-05-02 22:09:28</td>
      <td>345781</td>
      <td>view</td>
      <td>438400</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1464340</td>
      <td>2015-05-02 22:11:33</td>
      <td>345781</td>
      <td>addtocart</td>
      <td>438400</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1464573</td>
      <td>2015-05-02 22:15:30</td>
      <td>560305</td>
      <td>view</td>
      <td>43939</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1464908</td>
      <td>2015-05-02 22:23:57</td>
      <td>266417</td>
      <td>view</td>
      <td>445106</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Viewing a customer that purchased 
events[events['visitorid'] == random.choice(customer_purchased)]
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
      <th>timestamp</th>
      <th>visitorid</th>
      <th>event</th>
      <th>itemid</th>
      <th>transactionid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2082542</td>
      <td>2015-07-02 14:35:42</td>
      <td>1082371</td>
      <td>view</td>
      <td>418690</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2114941</td>
      <td>2015-07-03 14:20:34</td>
      <td>1082371</td>
      <td>view</td>
      <td>418690</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2114934</td>
      <td>2015-07-03 15:04:10</td>
      <td>1082371</td>
      <td>view</td>
      <td>418690</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2118279</td>
      <td>2015-07-03 15:04:32</td>
      <td>1082371</td>
      <td>view</td>
      <td>20218</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2111481</td>
      <td>2015-07-03 15:05:00</td>
      <td>1082371</td>
      <td>view</td>
      <td>409489</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2101701</td>
      <td>2015-07-03 15:05:28</td>
      <td>1082371</td>
      <td>view</td>
      <td>320377</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2111549</td>
      <td>2015-07-03 15:06:32</td>
      <td>1082371</td>
      <td>view</td>
      <td>94187</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2111597</td>
      <td>2015-07-03 15:07:25</td>
      <td>1082371</td>
      <td>view</td>
      <td>212289</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2124654</td>
      <td>2015-07-04 13:39:15</td>
      <td>1082371</td>
      <td>view</td>
      <td>418690</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2140678</td>
      <td>2015-07-05 13:47:17</td>
      <td>1082371</td>
      <td>view</td>
      <td>418690</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2149695</td>
      <td>2015-07-05 18:23:30</td>
      <td>1082371</td>
      <td>addtocart</td>
      <td>418690</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2149684</td>
      <td>2015-07-05 18:24:04</td>
      <td>1082371</td>
      <td>view</td>
      <td>418690</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2152288</td>
      <td>2015-07-05 18:43:38</td>
      <td>1082371</td>
      <td>transaction</td>
      <td>418690</td>
      <td>8178.0</td>
    </tr>
  </tbody>
</table>
</div>



### Did not result in transaction 


```python
## Customers that did not make a transaction
customer_not_purchased = [customer for customer in events['visitorid'].unique() if customer not in customer_purchased] ## Use not in to not double count customers that viewed and made transaction 

print(str(len(customer_not_purchased)) + (" customers did not make a purchase during ") + str((events['event'] != 'transaction').sum()) + " visits")
```

    1395861 customers did not make a purchase during 2733644 visits
    


```python
## Creating a dataframe of events for visitors that did not make a transaction
no_transaction_df = events[events['visitorid'].isin(customer_not_purchased)]
no_transaction_df.head()
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
      <th>timestamp</th>
      <th>visitorid</th>
      <th>event</th>
      <th>itemid</th>
      <th>transactionid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1462974</td>
      <td>2015-05-02 22:00:04</td>
      <td>693516</td>
      <td>addtocart</td>
      <td>297662</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1464806</td>
      <td>2015-05-02 22:00:11</td>
      <td>829044</td>
      <td>view</td>
      <td>60987</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1463000</td>
      <td>2015-05-02 22:00:13</td>
      <td>652699</td>
      <td>view</td>
      <td>252860</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1465287</td>
      <td>2015-05-02 22:00:24</td>
      <td>1125936</td>
      <td>view</td>
      <td>33661</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>1462955</td>
      <td>2015-05-02 22:00:26</td>
      <td>693516</td>
      <td>view</td>
      <td>297662</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Viewing a customer that did not make a purchase 
events[events['visitorid'] == random.choice(customer_not_purchased)]
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
      <th>timestamp</th>
      <th>visitorid</th>
      <th>event</th>
      <th>itemid</th>
      <th>transactionid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2194973</td>
      <td>2015-07-07 06:34:42</td>
      <td>1237506</td>
      <td>view</td>
      <td>23645</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Finding products that were bought together


```python
## Creating a dataframe of transactions
transaction_occured = transaction_df[transaction_df['event'] == 'transaction']
transaction_occured.head()
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
      <th>timestamp</th>
      <th>visitorid</th>
      <th>event</th>
      <th>itemid</th>
      <th>transactionid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1465072</td>
      <td>2015-05-02 22:27:21</td>
      <td>869008</td>
      <td>transaction</td>
      <td>40685</td>
      <td>9765.0</td>
    </tr>
    <tr>
      <td>1463096</td>
      <td>2015-05-02 22:35:01</td>
      <td>345781</td>
      <td>transaction</td>
      <td>438400</td>
      <td>1016.0</td>
    </tr>
    <tr>
      <td>1464289</td>
      <td>2015-05-02 23:01:47</td>
      <td>586756</td>
      <td>transaction</td>
      <td>440917</td>
      <td>10942.0</td>
    </tr>
    <tr>
      <td>1463462</td>
      <td>2015-05-02 23:07:38</td>
      <td>435495</td>
      <td>transaction</td>
      <td>175893</td>
      <td>6173.0</td>
    </tr>
    <tr>
      <td>1463605</td>
      <td>2015-05-02 23:31:14</td>
      <td>266417</td>
      <td>transaction</td>
      <td>445106</td>
      <td>12546.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Creating a list of lists of products that customers bought
purchased_together = []

for customer in customer_purchased:
    purchased_together.append(list(transaction_occured.loc[transaction_occured['visitorid'] == customer, 'itemid']))
```


```python
## Creating a function to find other items that customers purchased with a product. 
def productRecommender(itemid, purchased_together):
    recommended_products = []
    for item_list in purchased_together:
        if itemid in item_list:
            for item in item_list:
                if item != itemid:
                    recommended_products.append(item)
    return recommended_products
```


```python
## Products that were bought for product 179400
print("Products that were bought together with 179400: " + str(productRecommender(445106, purchased_together)))
```

    Products that were bought together with 179400: [301359, 238951, 117762]
    

## Creating a function to build a dataframe describing customer actions


```python
def creatingDataframe(customer_list):
    ## Building temperary df of customers events from customer list
    temp_df = events[events['visitorid'].isin(customer_list)]
    
    customer_df = pd.DataFrame()
    customer_df['visitorid'] = customer_list
    customer_df = customer_df.set_index('visitorid')
    
    #Calculating number of products the customer viewed
    num_products = temp_df.pivot_table(values = 'itemid', index = 'visitorid', aggfunc = lambda x: len(x.unique()))
    
    ## Calculating frequency of total customers views
    views = temp_df[temp_df['event'] == 'view'].groupby('visitorid').count()['event']
    
    ## Calculating frequency of a customers views
    added_to_cart = temp_df[temp_df['event'] == 'addtocart'].groupby('visitorid').count()['event']
    
    ## Calculating frequency of a customers transactions
    transaction_frequency = temp_df[temp_df['event'] == 'transaction'].groupby('visitorid').count()['event']
    
    
    
    
    customer_df = pd.concat([customer_df,
                             num_products,
                             views.rename('view_frequency'),
                             added_to_cart.rename('added_to_cart_frequency'),
                             transaction_frequency.rename('transaction_frequency')], axis=1, ignore_index = False)
    customer_df = customer_df.rename(columns = {'itemid' : 'num_products'})
    customer_df = customer_df.fillna(0)
    
    ## Returns a one if a customer made a purchase and a zero if no purchase was made
    customer_df['purchased'] = customer_df['transaction_frequency'].apply(lambda row: 1 if row > 0 else 0)
    return customer_df
    
```


```python
## Customer actions of a customer that made a purchase
purchasing_visitors = creatingDataframe(customer_purchased)
purchasing_visitors.head()
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
      <th>num_products</th>
      <th>view_frequency</th>
      <th>added_to_cart_frequency</th>
      <th>transaction_frequency</th>
      <th>purchased</th>
    </tr>
    <tr>
      <th>visitorid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>172</td>
      <td>22</td>
      <td>33.0</td>
      <td>3.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>186</td>
      <td>1</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>264</td>
      <td>2</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <td>419</td>
      <td>3</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <td>539</td>
      <td>1</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Customer actions of a customer that did not make a purchase
non_purchasing_visitors = creatingDataframe(customer_not_purchased)
non_purchasing_visitors.head()
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
      <th>num_products</th>
      <th>view_frequency</th>
      <th>added_to_cart_frequency</th>
      <th>transaction_frequency</th>
      <th>purchased</th>
    </tr>
    <tr>
      <th>visitorid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>3</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>4</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Randomizing and slicing the non_purchasing_visitors dataframe and combining it with the purchasing_visitors dataframe
non_purchasing_visitors = non_purchasing_visitors.sample(frac=1)[:28000]
full_customer_df = pd.concat([purchasing_visitors, non_purchasing_visitors], ignore_index=True)
full_customer_df = full_customer_df.sample(frac=1)
```

## Predicting if a visitor will make a purchase


```python
sns.pairplot(full_customer_df.drop('purchased', axis ='columns'))
```




    <seaborn.axisgrid.PairGrid at 0x23afc967288>




![png](output_37_1.png)



```python
full_customer_df.corr()
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
      <th>num_products</th>
      <th>view_frequency</th>
      <th>added_to_cart_frequency</th>
      <th>transaction_frequency</th>
      <th>purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>num_products</td>
      <td>1.000000</td>
      <td>0.990941</td>
      <td>0.859048</td>
      <td>0.873904</td>
      <td>0.098175</td>
    </tr>
    <tr>
      <td>view_frequency</td>
      <td>0.990941</td>
      <td>1.000000</td>
      <td>0.861749</td>
      <td>0.868861</td>
      <td>0.105714</td>
    </tr>
    <tr>
      <td>added_to_cart_frequency</td>
      <td>0.859048</td>
      <td>0.861749</td>
      <td>1.000000</td>
      <td>0.955225</td>
      <td>0.161880</td>
    </tr>
    <tr>
      <td>transaction_frequency</td>
      <td>0.873904</td>
      <td>0.868861</td>
      <td>0.955225</td>
      <td>1.000000</td>
      <td>0.178867</td>
    </tr>
    <tr>
      <td>purchased</td>
      <td>0.098175</td>
      <td>0.105714</td>
      <td>0.161880</td>
      <td>0.178867</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



We will only use the view_frequency in our prediction model. The other variables will be dropped because of multicollinearity. 


```python
data = ['view_frequency']
target = 'purchased'
```


```python
logreg = LogisticRegression(solver='lbfgs')
```


```python
## Splitting the data set 50/50 into a train and test dataset
train = full_customer_df[:27803]
test = full_customer_df[27803:]
```


```python
logreg.fit(train[data], train[target])
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
predictions = logreg.predict(test[data])
accuracy = metrics.accuracy_score(test[target], predictions) * 100
print("Prediction accuracy: " + str("{:.2f}".format(accuracy)) + "%")
```

    Prediction accuracy: 79.04%
    

# Takeaways

### We will be using the below image to aid us in making the proper decisions


```python
from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://i0.wp.com/www.digitalnoobs.com/wp-content/uploads/2015/09/buyer-decision-process.gif?resize=500%2C750")
```




<img src="https://i0.wp.com/www.digitalnoobs.com/wp-content/uploads/2015/09/buyer-decision-process.gif?resize=500%2C750"/>



The picture displays the 5 stages a buyer encounters in their buying decision process. The picutre is from: https://i0.wp.com/www.digitalnoobs.com/wp-content/uploads/2015/09/buyer-decision-process.gif?resize=500%2C750

### Visitor logs on and begins searching

When the visitor first logs on, we can recommend the most viewed products on the site from that week. 


```python
from datetime import timedelta
start_date = events['timestamp'][events.index[-1]]
end_date = start_date + timedelta(days=-7)
last_week = events[(events['timestamp'] >= end_date) & (events['timestamp'] <= start_date)]

featured_items = (last_week[last_week['event'] == 'view']['itemid'].value_counts())
print("This weeks most viewed items: ")
print(featured_items[:10])

print("\nWe will begin by showing the following items: " + str(list(featured_items.index)[:10]))
```

    This weeks most viewed items: 
    434782    419
    17114     384
    187946    250
    461686    171
    245085    153
    320130    132
    9877      108
    96924     102
    314579    102
    76060     101
    Name: itemid, dtype: int64
    
    We will begin by showing the following items: [434782, 17114, 187946, 461686, 245085, 320130, 9877, 96924, 314579, 76060]
    

### Visitor begins looking at itemid: 461686


```python
## Finding the most viewed related products
recommended_items = productRecommender(461686, purchased_together)
top_recommended = pd.Series(recommended_items).value_counts()
print("\nWe will show the following products that are commonly bought together with product 461686: " + str(list(top_recommended.index)[:10]))
```

    
    We will show the following products that are commonly bought together with product 461686: [119736, 10572, 171878, 32581, 218794, 320130, 248455, 357529, 420960, 124081]
    

### Visitor is on Stage 2 and begins looking at itemid: 10572


```python
print(category_ids.loc[[10572, 461686]])
```

           categoryid
    itemid           
    10572        1037
    461686       1037
    

We can see now that the visitor is looking at products within the same category, so the visitor is now moving to Stage 2. We will now recommend popular products from the category. 


```python
## Finding the most viewed products from categoryid 1037
currently_viewing = [10572]
merged_events = events.merge(right = category_ids, how = 'left', on ='itemid')
category_values = merged_events[merged_events['categoryid'] == '1037']
top_category_items = category_values[category_values['event'] == 'view']['itemid'].value_counts().index
top_items = [item for item in top_category_items if item not in currently_viewing][:10]
print("We will recommend the following products: " + str(top_items))
```

    We will recommend the following products: [461686, 422376, 218794, 171878, 32581, 271872, 67423, 285154, 307322, 75392]
    

### Visitor continues viewing products popular from the category and is moving on to Stage 3

If we had the available data, we would now recommend products within the same category that are the most similar in price. 

### At this point, the customer would have viewed atleast 4 items. We will look at how are strategy performed compared to our test dataset. 


```python
test['view_frequency'].describe()
```




    count    11916.000000
    mean         5.871182
    std         47.127793
    min          0.000000
    25%          1.000000
    50%          1.000000
    75%          3.000000
    max       2141.000000
    Name: view_frequency, dtype: float64



We will look at the distribution of the view frequencies and see if the mean is a good variable to compare our perfomance too.


```python
sns.kdeplot(test['view_frequency'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x23afd72a9c8>




![png](output_63_1.png)


The distribution is heavily skewed, so we will look at the median instead.


```python
print("The median of the data set is: " + str((test['view_frequency']).median()))
```

    The median of the data set is: 1.0
    

We can see above, that if our strategy performs as planned, it would result in getting more views from a visitor than 75% of visitors from the test dataframe, and we would have at least 3 more views than the median of the test dataframe. 


```python
## if the visitor has 4 views
four_views = test[test['view_frequency'] == 4]['purchased'].value_counts().loc[1] / test[test['view_frequency'] == 4]['purchased'].size * 100
print("From our test dataset, when a visitor had 4 views, the visitor made a transaction " + str("{:.2f}".format(four_views)) + "% of the time")

## if the visitor has 5 views
five_views = test[test['view_frequency'] == 5]['purchased'].value_counts().loc[1] / test[test['view_frequency'] == 5]['purchased'].size * 100
print("From our test dataset, when a visitor had 5 views, the visitor made a transaction " + str("{:.2f}".format(five_views)) + "% of the time")

## if the visitor has 6 views
six_views = test[test['view_frequency'] == 6]['purchased'].value_counts().loc[1] / test[test['view_frequency'] == 6]['purchased'].size * 100
print("From our test dataset, when a visitor had 6 views, the visitor made a transaction " + str("{:.2f}".format(six_views)) + "% of the time")
```

    From our test dataset, when a visitor had 4 views, the visitor made a transaction 51.53% of the time
    From our test dataset, when a visitor had 5 views, the visitor made a transaction 59.22% of the time
    From our test dataset, when a visitor had 6 views, the visitor made a transaction 63.87% of the time
    

## The strategy in action


```python
## Activity from visitor 1217315
events[events['visitorid'] == 1217315].merge(right = category_ids, how = 'left', on = 'itemid')
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
      <th>timestamp</th>
      <th>visitorid</th>
      <th>event</th>
      <th>itemid</th>
      <th>transactionid</th>
      <th>categoryid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>2015-09-15 09:46:37</td>
      <td>1217315</td>
      <td>view</td>
      <td>461686</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>1</td>
      <td>2015-09-15 09:47:10</td>
      <td>1217315</td>
      <td>view</td>
      <td>10572</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>2</td>
      <td>2015-09-15 09:49:59</td>
      <td>1217315</td>
      <td>view</td>
      <td>461686</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>3</td>
      <td>2015-09-15 09:50:25</td>
      <td>1217315</td>
      <td>view</td>
      <td>10572</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>4</td>
      <td>2015-09-15 09:54:36</td>
      <td>1217315</td>
      <td>view</td>
      <td>461686</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>5</td>
      <td>2015-09-15 09:56:08</td>
      <td>1217315</td>
      <td>addtocart</td>
      <td>461686</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>6</td>
      <td>2015-09-15 09:57:21</td>
      <td>1217315</td>
      <td>view</td>
      <td>32581</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>7</td>
      <td>2015-09-15 09:58:02</td>
      <td>1217315</td>
      <td>addtocart</td>
      <td>32581</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>8</td>
      <td>2015-09-15 10:01:13</td>
      <td>1217315</td>
      <td>transaction</td>
      <td>461686</td>
      <td>7909.0</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>9</td>
      <td>2015-09-15 16:21:06</td>
      <td>1217315</td>
      <td>view</td>
      <td>10572</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
    <tr>
      <td>10</td>
      <td>2015-09-16 00:00:17</td>
      <td>1217315</td>
      <td>view</td>
      <td>10572</td>
      <td>NaN</td>
      <td>1037</td>
    </tr>
  </tbody>
</table>
</div>



#### In the above dataframe, we can see an example of the scenario playing out. The visitor began by viewing a popular item, itemid 461686, clicked on a popular similar item, went back and forth between the two (Stage 2), looked at another similar item from the category (Stage 3), and ended up purchasing the product (Stage 4). 


```python

```
