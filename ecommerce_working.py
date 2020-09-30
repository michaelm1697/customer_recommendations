#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Importing needed libraries
import pandas as pd
import datetime
import numpy as np
import random
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[2]:


## Importing needed files
directory = 'C:/Users/viks9/Downloads/retailrocket/'

def importData(file):
    return pd.read_csv(directory + file + '.csv')

category_tree = importData('category_tree')
events = importData('events')
items1 = importData('item_properties_part1')
items2 = importData('item_properties_part2')
items_properties = pd.concat([items1, items2])


# ### Exploring and Cleaning Data

# In[3]:


category_tree.head()


# In[4]:


events.head()


# In[5]:


## Converting time stamp columns from UNIX format and sorting values by the timestamp
import datetime
events_times =[]
for i in events['timestamp']:
    events_times.append(datetime.datetime.fromtimestamp(i//1000.0)) 
events['timestamp'] = events_times

events = events.sort_values(by = 'timestamp')


# In[6]:


events.isnull().sum()


# We have 2733644 null values in the transactionid column, we will explore this more below.

# In[7]:


events[events['transactionid'].isnull()].head()


# In[8]:


events['event'].value_counts()


# In[9]:


## Calculating amount of transactions
non_transaction_events = events['event'].size - events['event'].value_counts()['transaction']
print("There were " + str(non_transaction_events) + " events that did not result in a transaction.")


# Null transaction ID's occur when an event is a view or addedtocart.

# In[10]:


## Calculating percentage of events that resulted in a transaction
percent_purchased = len(events[events['event'] == 'transaction']) / len(events[events['event'] != 'transaction']) * 100
print("Only " + str("{:.2f}".format(percent_purchased)) + "% of events resulted in a purchase")


# In[11]:


items_properties.head()


# The category tree dataframe does not have an itemid for us to reference, so we will get the categoryid's from the items_properties dataframe. When the property is categoryid, the value will be the items categoryid. 

# In[12]:


## Creating a dataframe for category ids
category_ids = items_properties[items_properties['property'] == 'categoryid']
category_ids = category_ids.rename(columns={"value": "categoryid"})
category_ids = category_ids.drop(['timestamp', 'property'], axis = 1)
category_ids = category_ids.set_index('itemid')
category_ids.head()


# In[13]:


## Converting time stamp columns from UNIX format and sorting values by the timestamp
items_times = []
for i in items_properties['timestamp']:
    items_times.append(datetime.datetime.fromtimestamp(i//1000.0)) 
items_properties['timestamp'] = items_times


# ## Splitting Events Dataframe for events that resulted in transaction and did not result in a transaction

# ### Resulted in a transaction
# 

# In[14]:


## Customers that made a transaction
customer_purchased = (events['visitorid'][events['event'] == 'transaction'].unique())

print(str(len(customer_purchased)) + (" customers made ") + str((events['event'] == 'transaction').sum()) + " purchases")


# In[15]:


## Creating a dataframe of the events for visitors that made a purchase
transaction_df = events[events['visitorid'].isin(customer_purchased)]
transaction_df.head()


# In[16]:


## Viewing a customer that purchased 
events[events['visitorid'] == random.choice(customer_purchased)]


# ### Did not result in transaction 

# In[17]:


## Customers that did not make a transaction
customer_not_purchased = [customer for customer in events['visitorid'].unique() if customer not in customer_purchased] ## Use not in to not double count customers that viewed and made transaction 

print(str(len(customer_not_purchased)) + (" customers did not make a purchase during ") + str((events['event'] != 'transaction').sum()) + " visits")


# In[18]:


## Creating a dataframe of events for visitors that did not make a transaction
no_transaction_df = events[events['visitorid'].isin(customer_not_purchased)]
no_transaction_df.head()


# In[19]:


## Viewing a customer that did not make a purchase 
events[events['visitorid'] == random.choice(customer_not_purchased)]


# ## Finding products that were bought together

# In[20]:


## Creating a dataframe of transactions
transaction_occured = transaction_df[transaction_df['event'] == 'transaction']
transaction_occured.head()


# In[21]:


## Creating a list of lists of products that customers bought
purchased_together = []

for customer in customer_purchased:
    purchased_together.append(list(transaction_occured.loc[transaction_occured['visitorid'] == customer, 'itemid']))


# In[22]:


## Creating a function to find other items that customers purchased with a product. 
def productRecommender(itemid, purchased_together):
    recommended_products = []
    for item_list in purchased_together:
        if itemid in item_list:
            for item in item_list:
                if item != itemid:
                    recommended_products.append(item)
    return recommended_products


# In[23]:


## Products that were bought for product 179400
print("Products that were bought together with 179400: " + str(productRecommender(445106, purchased_together)))


# ## Creating a function to build a dataframe describing customer actions

# In[24]:


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
    


# In[25]:


## Customer actions of a customer that made a purchase
purchasing_visitors = creatingDataframe(customer_purchased)
purchasing_visitors.head()


# In[26]:


## Customer actions of a customer that did not make a purchase
non_purchasing_visitors = creatingDataframe(customer_not_purchased)
non_purchasing_visitors.head()


# In[27]:


## Randomizing and slicing the non_purchasing_visitors dataframe and combining it with the purchasing_visitors dataframe
non_purchasing_visitors = non_purchasing_visitors.sample(frac=1)[:28000]
full_customer_df = pd.concat([purchasing_visitors, non_purchasing_visitors], ignore_index=True)
full_customer_df = full_customer_df.sample(frac=1)


# ## Predicting if a visitor will make a purchase

# In[28]:


sns.pairplot(full_customer_df.drop('purchased', axis ='columns'))


# In[29]:


full_customer_df.corr()


# We will only use the view_frequency in our prediction model. The other variables will be dropped because of multicollinearity. 

# In[30]:


data = ['view_frequency']
target = 'purchased'


# In[31]:


logreg = LogisticRegression(solver='lbfgs')


# In[32]:


## Splitting the data set 50/50 into a train and test dataset
train = full_customer_df[:27803]
test = full_customer_df[27803:]


# In[33]:


logreg.fit(train[data], train[target])


# In[34]:


predictions = logreg.predict(test[data])
accuracy = metrics.accuracy_score(test[target], predictions) * 100
print("Prediction accuracy: " + str("{:.2f}".format(accuracy)) + "%")


# # Takeaways

# ### We will be using the below image to aid us in making the proper decisions

# In[35]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://i0.wp.com/www.digitalnoobs.com/wp-content/uploads/2015/09/buyer-decision-process.gif?resize=500%2C750")


# The picture displays the 5 stages a buyer encounters in their buying decision process. The picutre is from: https://i0.wp.com/www.digitalnoobs.com/wp-content/uploads/2015/09/buyer-decision-process.gif?resize=500%2C750

# ### Visitor logs on and begins searching

# When the visitor first logs on, we can recommend the most viewed products on the site from that week. 

# In[36]:


from datetime import timedelta
start_date = events['timestamp'][events.index[-1]]
end_date = start_date + timedelta(days=-7)
last_week = events[(events['timestamp'] >= end_date) & (events['timestamp'] <= start_date)]

featured_items = (last_week[last_week['event'] == 'view']['itemid'].value_counts())
print("This weeks most viewed items: ")
print(featured_items[:10])

print("\nWe will begin by showing the following items: " + str(list(featured_items.index)[:10]))


# ### Visitor begins looking at itemid: 461686

# In[37]:


## Finding the most viewed related products
recommended_items = productRecommender(461686, purchased_together)
top_recommended = pd.Series(recommended_items).value_counts()
print("\nWe will show the following products that are commonly bought together with product 461686: " + str(list(top_recommended.index)[:10]))


# ### Visitor is on Stage 2 and begins looking at itemid: 10572

# In[38]:


print(category_ids.loc[[10572, 461686]])


# We can see now that the visitor is looking at products within the same category, so the visitor is now moving to Stage 2. We will now recommend popular products from the category. 

# In[39]:


## Finding the most viewed products from categoryid 1037
currently_viewing = [10572]
merged_events = events.merge(right = category_ids, how = 'left', on ='itemid')
category_values = merged_events[merged_events['categoryid'] == '1037']
top_category_items = category_values[category_values['event'] == 'view']['itemid'].value_counts().index
top_items = [item for item in top_category_items if item not in currently_viewing][:10]
print("We will recommend the following products: " + str(top_items))


# ### Visitor continues viewing products popular from the category and is moving on to Stage 3

# If we had the available data, we would now recommend products within the same category that are the most similar in price. 

# ### At this point, the customer would have viewed atleast 4 items. We will look at how are strategy performed compared to our test dataset. 

# In[40]:


test['view_frequency'].describe()


# We will look at the distribution of the view frequencies and see if the mean is a good variable to compare our perfomance too.

# In[41]:


sns.kdeplot(test['view_frequency'])


# The distribution is heavily skewed, so we will look at the median instead.

# In[42]:


print("The median of the data set is: " + str((test['view_frequency']).median()))


# We can see above, that if our strategy performs as planned, it would result in getting more views from a visitor than 75% of visitors from the test dataframe, and we would have at least 3 more views than the median of the test dataframe. 

# In[43]:


## if the visitor has 4 views
four_views = test[test['view_frequency'] == 4]['purchased'].value_counts().loc[1] / test[test['view_frequency'] == 4]['purchased'].size * 100
print("From our test dataset, when a visitor had 4 views, the visitor made a transaction " + str("{:.2f}".format(four_views)) + "% of the time")

## if the visitor has 5 views
five_views = test[test['view_frequency'] == 5]['purchased'].value_counts().loc[1] / test[test['view_frequency'] == 5]['purchased'].size * 100
print("From our test dataset, when a visitor had 5 views, the visitor made a transaction " + str("{:.2f}".format(five_views)) + "% of the time")

## if the visitor has 6 views
six_views = test[test['view_frequency'] == 6]['purchased'].value_counts().loc[1] / test[test['view_frequency'] == 6]['purchased'].size * 100
print("From our test dataset, when a visitor had 6 views, the visitor made a transaction " + str("{:.2f}".format(six_views)) + "% of the time")


# ## The strategy in action

# In[44]:


## Activity from visitor 1217315
events[events['visitorid'] == 1217315].merge(right = category_ids, how = 'left', on = 'itemid')


# #### In the above dataframe, we can see an example of the scenario playing out. The visitor began by viewing a popular item, itemid 461686, clicked on a popular similar item, went back and forth between the two (Stage 2), looked at another similar item from the category (Stage 3), and ended up purchasing the product (Stage 4). 
