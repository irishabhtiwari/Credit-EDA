#!/usr/bin/env python
# coding: utf-8

# ## Reading and Understanding the Given Data

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Reading the Application Data

# In[3]:


bank = pd.read_csv('application_data.csv')


# In[4]:


bank.head()


# In[5]:


bank.describe()


# In[6]:


bank.info('all')


# In[7]:


bank.shape


# ## DEALING WITH NULL VALUES

# In[8]:


#null value percentage in respective columns
bank_null = bank.isnull().mean()*100
bank_null


# In[9]:


#  number of columns having null values more than 30%

bank_null = bank.isnull().sum()
bank_null = bank_null[bank_null > 0.3*(len(bank_null))]
len(bank_null)


# In[10]:


# Dropping the above columns

bank_null = list(bank_null[bank_null.values>=0.3].index)
bank.drop(labels = bank_null, axis =1,inplace = True)


# In[11]:


# Quick check after dropping the null values

bank.isnull().sum()/len(bank)*100


# #### Insights 
# - given data has 307511 rows and 122 columns.
# - It contains float64(65), int64(41), object(16).
# - There are 64 columns with null values more than 30%.

# In[12]:


bank.isnull().sum()


#  #### Insight
#  - AMT_ANNUITY has 12 null values, CNT_FAM_MEMBERS has 2 null values, DAYS_LAST_PHONE_CHANGE has 1 null value.

# In[13]:


# now let's check the rowsa and remove all the null rows having null values >= 30%

blank_row=bank.isnull().sum(axis=1)
blank_row=list(blank_row[blank_row.values>=0.3*len(bank)].index)
bank.drop(labels=blank_row,axis=0,inplace=True)
print(len(blank_row))


# #### Important Observation
# - There are some of the columns where the value as 'XNA' which means 'Not Available'
# - So, now we have to find the respective rows and columns and implement appropiate techniques on them to fill those missing values or to delete them.

# In[14]:


# let's first deal with 'XNA' values in columns

# counting 'XNA' in organization type column

bank['ORGANIZATION_TYPE'].value_counts()


# In[15]:


# counting total value in organization
bank['ORGANIZATION_TYPE'].describe()


# In[16]:


percentage_of_null_org_type = (55374/307511)*100
percentage_of_null_org_type


# #### Insight
# - 'ORGANIZATION_TYPE', we have total count of 307511 rows of which 55374 rows are having 'XNA' values which is 18%
# - So, dropping the 'XNA' in 'ORGANIZATION_TYPE' won't have any major impact on this dataset.
# 

# In[17]:


# dropping 'XNA' in 'ORGANIZATION_TYPE'

bank=bank.drop(bank.loc[bank['ORGANIZATION_TYPE']=='XNA'].index)
bank[bank['ORGANIZATION_TYPE']=='XNA'].shape


# In[22]:


# Analysing 'CODE_GENDER' for 'XNA'
bank['CODE_GENDER'].value_counts()


# #### Insight
# - There are 4 'XNA' in 'CODE_GENDER'
# - As majority of the values are 'F' we can update null values as 'F' as it won't have any major impact.
# 

# In[23]:


# updating 'CODE_GENDER' as 'F'
bank.loc[bank['CODE_GENDER']=='XNA','CODE_GENDER']='F'
bank['CODE_GENDER'].value_counts()


# In[25]:


palette_color = sns.color_palette('bright')
keys = ['Female', 'Male']
# plotting data on chart
plt.pie(bank['CODE_GENDER'].value_counts()
, labels=keys, colors=palette_color, autopct='%.0f%%')
  
# displaying chart
plt.show()


# In[26]:


# Casting all variable into numeric in the dataset

numeric_cols=['TARGET','CNT_CHILDREN','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','REGION_POPULATION_RELATIVE','DAYS_BIRTH',
                'DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','HOUR_APPR_PROCESS_START','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

bank[numeric_cols]=bank[numeric_cols].apply(pd.to_numeric)
bank.head(5)


# In[27]:


# lets analyse and remove the unwanted rows for furhter analysis and drop the same

unwanted_cols = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY','FLAG_EMAIL','CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY','DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']

bank.drop(labels=unwanted_cols,axis=1,inplace=True)


# #### Creating Bins
# - Creating bins for continous variable categories column 'AMT_INCOME_TOTAL' and 'AMT_CREDIT' for better analysis.

# In[28]:


# Creating bins for Credit amount.

bins = [0,150000,200000,250000,300000,350000,400000,450000,500000,550000,600000,650000,700000,750000,800000,850000,900000,1000000000]
slots = ['0-150000', '150000-200000','200000-250000', '250000-300000', '300000-350000', '350000-400000','400000-450000',
        '450000-500000','500000-550000','550000-600000','600000-650000','650000-700000','700000-750000','750000-800000',
        '800000-850000','850000-900000','900000 and above']

bank['AMT_CREDIT_RANGE']=pd.cut(bank['AMT_CREDIT'],bins=bins,labels=slots)


# In[29]:


# Creating bins for income amount.

bins = [0,25000,50000,75000,100000,125000,150000,175000,200000,225000,250000,275000,300000,325000,350000,375000,400000,425000,450000,475000,500000,10000000000]
slot = ['0-25000', '25000-50000','50000-75000','75000,100000','100000-125000', '125000-150000', '150000-175000','175000-200000',
       '200000-225000','225000-250000','250000-275000','275000-300000','300000-325000','325000-350000','350000-375000',
       '375000-400000','400000-425000','425000-450000','450000-475000','475000-500000','500000 and above']

bank['AMT_INCOME_RANGE']=pd.cut(bank['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[30]:


# Dividing the dataset into two dataset of  target=1(client with payment difficulties) and target=0(all other)

target0=bank.loc[bank["TARGET"]==0]
target1=bank.loc[bank["TARGET"]==1]


# In[31]:


# Calculating Imbalance percentage
    
# Since the majority is target0 and minority is target1

round(len(target0)/len(target1),2)


# ## UNIVARIATE ANALYSIS

# #### FOR TARGET = 0

# In[32]:


# Count plotting in logarithmic scale

def splot(df,col,title,hue =None):
    
    sns.set_style('darkgrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 22
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.titlepad'] = 32
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='ocean') 
        
    plt.show()


# In[72]:


# plotting for income range
splot(target0,col='AMT_INCOME_RANGE',title='Distribution of income range',hue='CODE_GENDER')


# #### Insights
# - Female counts are higher than male.
# - Income ranging 100000 to 200000 owns large number of credits.
# - Very less count for income ranging 400000 and above.
# - This graph shows that females are more than male in having credits for the income ranging 100000 to 200000.

# In[34]:


# Plotting for Contract type

splot(target0,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')


# #### Insights
# - For contract type ‘cash loans’ is having higher number of credits than ‘Revolving loans’ contract type.
# - Females are leading for applying credits for both in Cash loans and Revolving loans.

# In[35]:


# Count Plotting for Income type

splot(target0,col='NAME_INCOME_TYPE',title='Distribution of Income type',hue='CODE_GENDER')


# #### Insights
# - For income type ‘working’, ’commercial associate’, and ‘State Servant’, credits are maximum.
# - Females are having more credits than male.
# - Lesser credits for income type ‘student’ ,’pensioner’, ‘Businessman’ and ‘Maternity leave’.

# In[36]:


# Organization type Plotting for logarithmic scale

sns.set_style('dark')
sns.set_context('talk')
plt.figure(figsize=(15,30))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30

plt.title("Distribution of Organization type for target - 0")

plt.xticks(rotation=90)
plt.xscale('log')

sns.countplot(data=target0,y='ORGANIZATION_TYPE',order=target0['ORGANIZATION_TYPE'].value_counts().index,palette='viridis')

plt.show()


# #### Insights
# - Most of the clients which have applied for credits are from ‘Business entity Type 3’ , ‘Self employed’, ‘Other’ , ‘Medicine’ and ‘Government’.
# - Less clients are from Industry type 8, Indsutry : type13, tarde : type 4, religion and Indsutry : type10.

# #### FOR TARGET = 1

# In[37]:


# plotting for income range

splot(target1,col='AMT_INCOME_RANGE',title='Distribution of income range',hue='CODE_GENDER')


# #### Insights
# - Male counts are higher than female.
# - Income ranging 100000 to 200000 owns large number of credits.
# - Very less count for income ranging 400000 and above.
# - This graph shows that males are more than females in having credits for the income ranging 100000 to 200000.

# In[38]:


# Plotting for Contract type

splot(target1,col='NAME_CONTRACT_TYPE',title='Distribution of contract type',hue='CODE_GENDER')


# #### Insights
# - For contract type ‘cash loans’ is having higher number of credits than ‘Revolving loans’ contract type.
# - Females are leading for applying credits for both in Cash loans and Revolving loans.

# In[39]:


# Count Plotting for Income type

splot(target1,col='NAME_INCOME_TYPE',title='Distribution of Income type',hue='CODE_GENDER')


# #### Insights
# - For income type ‘working’, ’commercial associate’, and ‘State Servant’, credits are maximum.
# - Females are having more credits than male.
# - Lesser credits for income type ‘Maternity leave’.
# - For type 1: There is no income type for ‘student’ , ’pensioner’ and ‘Businessman’.So, we can conclude that they don’t do late payments.

# In[40]:


# Organization type Plotting for logarithmic scale

sns.set_style('dark')
sns.set_context('talk')
plt.figure(figsize=(17,32))
plt.rcParams["axes.labelsize"] = 20
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['axes.titlepad'] = 30

plt.title("Distribution of Organization type for target - 1")

plt.xticks(rotation=90)
plt.xscale('log')

sns.countplot(data=target1,y='ORGANIZATION_TYPE',order=target1['ORGANIZATION_TYPE'].value_counts().index,palette='viridis')

plt.show()


# #### Insights
# - Most of the clients which have applied for credits are from ‘Business entity Type 3’ , ‘Self employed’, ‘Other’ , ‘Business entity Type 2’ and ‘Construction’.
# - Less clients are from Industry type 4,type 8, type 5, religion and industry : type10, industry : type6.

# ### CORRELATION
# 

# In[41]:


# Finding some correlation for numerical columns for both target 0 and 1 

target0_cor=target0.iloc[0:,2:]
target1_cor=target1.iloc[0:,2:]

target0_c=target0_cor.corr(method='spearman')
target1_c=target1_cor.corr(method='spearman')


# In[42]:


# correlation for target0
target0


# In[43]:


#correlation for target1
target1


# In[44]:


# Now, let's plot the above correlation with heat map as it is the best choice to visulaize

def targets_corr(data,title):
    plt.figure(figsize=(15, 10))
    plt.rcParams['axes.titlesize'] = 27
    plt.rcParams['axes.titlepad'] = 72

# heatmap with a color map of choice

    sns.heatmap(data, cmap="RdYlGn",annot=False)

    plt.title(title)
    plt.yticks(rotation=0)
    plt.show()


# In[45]:


# For Target 0

targets_corr(data=target0_c,title='Correlation for target 0')


# #### Insights
# - Credit amount is inversely proportional to the date of birth and number of children client have which means Credit amount is higher for low age and Credit amount is higher for less children count client have
# - Income amount is inversely proportional to the number of children client have which means more income for lesser children clients we have
# - Densely populated are has lesser children client.
# - The income is higher in densely populated area.
# - Credit amount is higher in densely populated area.

# In[46]:


# For Target 1

targets_corr(data=target1_c,title='Correlation for target 1')


# #### Insights
# - The client's permanent address does not match contact address are having less children and vice-versa.
# - The client's permanent address does not match work address are having less children and vice-versa.
# - Rest of the obeservations are similar to target0.

# ### Analysis of different variables

# In[47]:


# Box plotting for univariate variables analysis in logarithmic scale

def univariate_num(data,col,title):
    sns.set_style('dark')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 22
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['axes.titlepad'] = 32
    
    plt.title(title)
    plt.yscale('log')
    sns.boxplot(data =data, y=col,orient='v',palette = "colorblind")
    plt.show()


# #### For target 0

# In[48]:


# Distribution of income amount

univariate_num(data=target0,col='AMT_INCOME_TOTAL',title='Distribution of income amount')


# #### Insights
# - There are some outliers in income amount.
# - The third quartiles is very sleak for income amount.
# 

# In[49]:


# Disrtibution of credit amount

univariate_num(data=target0,col='AMT_CREDIT',title='Distribution of credit amount')


# #### Insights
# - There are some outliers in credit amount.
# - Most of the credits of clients are in frist quartile as we can see that first quartile is bigger than third quartile.

# In[50]:


# Distribution of anuuity amount

univariate_num(data=target0,col='AMT_ANNUITY',title='Distribution of Annuity amount')


# #### Insights
# - There are some outliers in annuity amount.
# - Most of the annuity clients are in frist quartile as we can see that first quartile is bigger than third quartile.

# #### For target 1

# In[51]:


# Distribution of income amount

univariate_num(data=target1,col='AMT_INCOME_TOTAL',title='Distribution of income amount')


# #### Insights
# - There are some outliers in income amount.
# - The third quartiles is very slim for income amount.
# - Most of the income clients are present in first quartile.

# In[52]:


# Distribution of credit amount

univariate_num(data=target1,col='AMT_CREDIT',title='Distribution of credit amount')


# #### Insights
# - There are some outliers in credit amount.
# - Most of the credits of clients are in frist quartile as we can see that first quartile is bigger than third quartile.

# In[53]:


# Distribution of Annuity amount

univariate_num(data=target1,col='AMT_ANNUITY',title='Distribution of Annuity amount')


# #### Insights
# - There are some outliers in credit amount.
# - Most of the annuity clients are in frist quartile as we can see that first quartile is bigger than third quartile.

# ## BIVARIATE ANALYSIS

# #### For target 0

# In[54]:


# Box plotting for Income amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =target0, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v', palette = "bright")
plt.title('Income amount vs Education Status')
plt.show()


# #### Insights
# - For Education type 'Higher education' the income amount is mostly same across different family status.
# - It does contain many outliers. 
# - Academic degree has the least number of outliers but their income amount is little higher than 'Higher education'. 
# - Lower secondary of civil marriage family is having the least income amount.

# In[55]:


# Box plotting for Credit amount

plt.figure(figsize=(16,12))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =target0, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v',palette = "bright")
plt.title('Credit amount vs Education Status')
plt.show()


# #### Insights
# - Family status 'civil marriage', 'married' and 'separated' in 'Academic degree' education has the highest number of credits.
# - Higher the education of family status of 'marriage', 'single' and 'civil marriage', higher is the number of outliers.
# - Civil marriage in Academic degree is having majority of the credits in the third quartile.

# #### For Target 1

# In[56]:


# Box plotting for Income amount in logarithmic scale

plt.figure(figsize=(18,14))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =target1, x='NAME_EDUCATION_TYPE',y='AMT_INCOME_TOTAL', hue ='NAME_FAMILY_STATUS',orient='v', palette = "bright")
plt.title('Income amount vs Education Status')
plt.show()


# #### Insights
# - For 'higher education' income is mostly same across different family status.
# - It does contain many outliers.
# - Academic degree has the least number of outliers but the income of all other family status than 'Married' have negligeble income
# - For family status 'Married', income is significantly higher in 'Academic dgree'. 
# - Lower secondary of 'Widow' family is having the least income.

# In[57]:


# Box plotting for Credit amount

plt.figure(figsize=(16,11))
plt.xticks(rotation=45)
plt.yscale('log')
sns.boxplot(data =target1, x='NAME_EDUCATION_TYPE',y='AMT_CREDIT', hue ='NAME_FAMILY_STATUS',orient='v',palette = "bright")
plt.title('Credit amount vs Education Status')
plt.show()


# #### Insights
# - 'Married' in 'Academic degree' education has the highest credit amount.
# - Most of the outliers are from Education type 'Higher education' and 'Secondary'.
# - 'Widow' for 'Lower secondary' has most of it's credit in third quartile.

# ## Reading the dataset of previous application

# In[58]:


bank1 = pd.read_csv("previous_application.csv")


# In[59]:


bank1.head()


# #### Cleaning the missing values

# In[60]:


# finding null values

nullcol0=bank1.isnull().sum()
nullcol0=nullcol0[nullcol0.values>(0.3*len(nullcol0))]
len(nullcol0)


# In[61]:


# Removing those 15 columns

nullcol0 = list(nullcol0[nullcol0.values>=0.3].index)
bank1.drop(labels=nullcol0,axis=1,inplace=True)


# In[62]:


# Removing those 15 columns
bank1.shape


# In[63]:


# Removing the column values of 'XNA' and 'XAP' from bank1

bank1=bank1.drop(bank1[bank1['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
bank1=bank1.drop(bank1[bank1['NAME_CASH_LOAN_PURPOSE']=='XNA'].index)
bank1=bank1.drop(bank1[bank1['NAME_CASH_LOAN_PURPOSE']=='XAP'].index)

bank1.shape


# In[64]:


# Now merging the Application dataset with previous appliaction dataset

new_bank=pd.merge(left=bank,right=bank1,how='inner',on='SK_ID_CURR',suffixes='_x')


# In[65]:


# Renaming the column names after merging

new_bank1 = new_bank.rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',
                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',
                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',
                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',
                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',
                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1).rename({'NAME_CONTRACT_TYPE_' : 'NAME_CONTRACT_TYPE','AMT_CREDIT_':'AMT_CREDIT','AMT_ANNUITY_':'AMT_ANNUITY',
                         'WEEKDAY_APPR_PROCESS_START_' : 'WEEKDAY_APPR_PROCESS_START',
                         'HOUR_APPR_PROCESS_START_':'HOUR_APPR_PROCESS_START','NAME_CONTRACT_TYPEx':'NAME_CONTRACT_TYPE_PREV',
                         'AMT_CREDITx':'AMT_CREDIT_PREV','AMT_ANNUITYx':'AMT_ANNUITY_PREV',
                         'WEEKDAY_APPR_PROCESS_STARTx':'WEEKDAY_APPR_PROCESS_START_PREV',
                         'HOUR_APPR_PROCESS_STARTx':'HOUR_APPR_PROCESS_START_PREV'}, axis=1)


# In[66]:


# Removing unwanted columns for analysis

new_bank1.drop(['SK_ID_CURR','WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION', 
              'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
              'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY','WEEKDAY_APPR_PROCESS_START_PREV',
              'HOUR_APPR_PROCESS_START_PREV', 'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY'],axis=1,inplace=True)


# In[67]:


new_bank1.shape


# ## UNIVARIATE ANALYSIS

# In[68]:


# Distribution of contract status in logarithmic scale

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(17,32))
plt.rcParams["axes.labelsize"] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.titlepad'] = 32
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of contract status with purposes')
ax = sns.countplot(data = new_bank1, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=new_bank1['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'NAME_CONTRACT_STATUS',palette='viridis') 


# #### Insights
# - Most rejection of loans came from purpose 'repairs'.
# - For education purpose we have equal number of approves and rejections.
# - Paying other loans and buying a new car has significantly higher rejection than approves.

# In[69]:


# Distribution of contract status

sns.set_style('whitegrid')
sns.set_context('talk')

plt.figure(figsize=(17,32))
plt.rcParams["axes.labelsize"] = 22
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.titlepad'] = 32
plt.xticks(rotation=90)
plt.xscale('log')
plt.title('Distribution of purposes with target ')
ax = sns.countplot(data = new_bank1, y= 'NAME_CASH_LOAN_PURPOSE', 
                   order=new_bank1['NAME_CASH_LOAN_PURPOSE'].value_counts().index,hue = 'TARGET',palette='viridis') 


# #### Insights
# - Loan purposes with 'Repairs' are facing more difficulites in payment on time.
# - There are few places where loan payment is significantly higher than facing difficulties. They are 'Buying a garage', 'Business developemt', 'Buying land','Buying a new car' and 'Education'.
# 
# **Hence we should focus on these purposes as these clients are having for minimal payment difficulties.**

# ## BIVARIATE ANALYSIS

# In[70]:


# Box plotting for Credit amount in logarithmic scale

plt.figure(figsize=(16,12))
plt.xticks(rotation=90)
plt.yscale('log')
sns.boxplot(data =new_bank1, x='NAME_CASH_LOAN_PURPOSE',hue='NAME_INCOME_TYPE',y='AMT_CREDIT_PREV',orient='v', palette = "bright")
plt.title('Prev Credit amount vs Loan Purpose')
plt.show()


# #### Insights
# - The credit amount for Loan purposes like 'Buying a home','Buying a land','Buying a new car' and'Building a house' is higher.
# - Income type of 'state servants' have a significant amount of credit applied.
# - Money for 'third person' or a' Hobby' is having lesser credits.

# In[71]:


# Box plotting for Credit amount prev vs Housing type in logarithmic scale

plt.figure(figsize=(18,14))
plt.xticks(rotation=90)
sns.barplot(data =new_bank1, y='AMT_CREDIT_PREV',hue='TARGET',x='NAME_HOUSING_TYPE',palette = "ocean")
plt.title('Prev Credit amount vs Housing type')
plt.show()


# #### Insights
# - For Housing type, office appartment is having higher credit of target 0 and 'co-op apartment' is having higher credits for target 1.
# - Bank should avoid giving loans to the housing type of 'co-op apartment' as they are having difficulties in payment.
# - Bank can focus mostly on housing type 'with parents' or 'House\appartment' or 'miuncipal appartment' for successful payments.

# # CONCLUSION

# - **Banks should focus more on contract type ‘Student’ ,’pensioner’ and ‘Businessman’ with housing ‘type other than ‘Co-op apartment’ for successful payments.**
# 
# - **Banks should focus less on income type ‘Working’ as they are having most number of unsuccessful payments.**
# 
# - **Also with loan purpose ‘Repair’ is having higher number of unsuccessful payments on time.**
# 
# - **We should be more focused targeting clients from housing type ‘With parents’ as they are having least number of unsuccessful payments.**
