#!/usr/bin/env python
# coding: utf-8

# <title>Titanic DataSet: Data Analysis</title>

# <h1>Data Analysis Project :Titanic DataSet</h1>

# <h3>By Gokula Chandra.K ((21.05.2021))</h3>

# <p> In this worksheet , we will analyse the famous dataset on kaggle<br>
# And also on survival chances of passengers on multiple factors<br>
# we will answer the following questions<br>
# Who were the passengers on the Titanic? (based on categories like age, gender, class.. etc)<br>
# What deck were the passengers on and how does that relate to their class?<br>
# Where did the passengers come from?(City)<br>
# What is the relation between survival chance and passenger travelling status?(Alone or with Family)<br>
# What are the factors affected a passenger in the process of survival in the disaster?<br>
# <br> Importing the required packages..........<br> 
# </p>

# import pandas as pd
# import numpy as np
# from pandas import Series,DataFrame
# 
# import matplotlib.pyplot as plt
# import seaborn as sns

# <p>
# The dataset related to titanic information is downloaded from kaggle<br>
# Reading the dataset using read_csv()<br>
# </p>

# In[148]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train=pd.read_csv('train.csv')
titanic_df = pd.read_csv('tested.csv')


# In[88]:


titanic_df.head()


# <p><b>An Complete info regarding the entries in the dataset along with the data types including their count <br>
#    Other attributes of intervals at 25,50,75 and max along with standard dev. of those values (Complete description)<br>
#     </b></p>

# In[89]:


titanic_df.info()


# In[90]:


titanic_df.describe(include='all')


# <p><b>There are various plots in seaborn module one of such is factor plot but it is updated in present versions hence</b><br>
#     <b>we use catplot in its place to visualize the dataframe in form of a graph with axes metrics</b></p>

# In[91]:


sns.factorplot('Sex',data=titanic_df,kind='count')


# <p><b>Visual representation of passengers based on Gender parameters</b></p>

# In[92]:


sns.catplot(x='Sex',kind='count',data=titanic_df)


# <p><b>
#     Ostentatiously , we see male count is more than female count as per above graph<br>
#     Lets visualize on basis of Pclass attribute to have an unvague view:<br>
# </b></p>

# In[93]:


sns.catplot(x='Pclass',data=titanic_df,hue='Sex',kind='count')


# <p><b> Now, we try adding a column named Person in the titanic data frame by checking it with the 
#     condition as with age and also we use apply method to do so,<br>
#     WE Classify them as Child if there age falls below 16 else remain unchanged<br></b>
# </p>

# In[94]:


def mwc(passenger):
    age=passenger['Age']
    sex=passenger['Sex']
    return 'child' if age<16 else sex
titanic_df['Person'] = titanic_df.apply(mwc,axis=1)


# In[95]:


titanic_df[0:100]


# <p><b>
#     Now, we see survival chances among the Person class (Distribution chart)
#     </b></p>

# In[96]:


p_sur=titanic_df.groupby(['Person','Survived'])
p_sur.describe()


# In[97]:


sns.catplot(x='Person',data=titanic_df,hue='Sex',kind='count')


# <p><b>Lets have an count of all categories of persons along their respective counts</b><br>
#     <b> And also lets have a survival distribution between male and female</b></p>

# In[153]:


def bar_plot(feature):
    survive = train[train['Survived']==1][feature].value_counts()
    dead    = train[train['Survived']==0][feature].value_counts()
    tf=pd.DataFrame([survive,dead])
    tf.index=['Survived','Dead']
    tf.plot(kind='bar',stacked=True,figsize=(10,5))
bar_plot('Sex')
bar_plot('Pclass')
bar_plot('SibSp')


# In[98]:


print(titanic_df['Person'].value_counts())


# <p><b>
#     Now, we plot the survival rates against every Person class
#     </b></p>

# In[99]:


sns.catplot(x='Pclass',data=titanic_df,hue='Person',kind='count')


# <p><b> Histogram of ages will give more general perspective on the ages of the persons..</b></p>

# In[100]:


titanic_df['Age'].hist()


# <p><b>
#     Now, we have FacetGrid from seaborn module which facilitate us to have wider view of distributions along with<br>
#     regions under the curve as shaded and also kdeplot will let us know local minima and maxima of the distribution<br>
#     <br>
#     here, we plot them against the Pclass and Sex parameters<br>
#     </b></p>

# In[101]:


fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
#fig.map(sns.scatterplot,"Age","Person")


# In[102]:


fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
fig.set(xlim=(0,90))
fig.add_legend()


# <p><b> 
#     We have certain range of palette to better locate the components of the curve along with continous probability<br>
#     density curve as analagous to histogram</b></p>

# In[103]:


sns.color_palette("Set3")


# In[104]:


sns.color_palette("Set2")


# In[105]:


sns.color_palette("Set2")
fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)
fig.map(sns.kdeplot,'Age',shade=True)
fig.set(xlim=(0,90))
fig.add_legend()


# In[106]:


sns.color_palette("flare",as_cmap=True)


# <h2>Now,we are trying to find the distribution and Analysis of passengers based on Decks</h2>

# <p><b> We filter the Null values reffered as not assigned values inorder to analyze the passengers<br>
#     survived based on Decks<br></b></p>

# In[107]:


deck_df = titanic_df.dropna(axis=0)
deck_df.head()


# In[108]:


deck_df.info()


# <p> <b>Here, we classify based on decks:</b></p>

# In[109]:


def get_level(passenger):
    cabin = passenger['Cabin']
    return cabin[0]
deck_df['level']=deck_df.apply(get_level,axis=1)


# In[110]:


deck_df.head()


# In[111]:


sns.catplot(x='level',data=deck_df,palette='spring_d',kind='count')


# In[112]:


sns.catplot(x='level',data=deck_df,palette='winter_d',hue='Pclass',kind='count')


# In[113]:


sns.catplot(x='Embarked',data=deck_df,hue='Pclass',kind='count')


# 
# <h2>We notice that Decks(A,B,C,D,E) related to class 1 and Decks(D,F) related to class 2 and Decks(F,G) related to class3</h2>
# <p><b>
# We will find the distribution of passengers based on the embarked cities<br>
#     &emsp;&emsp;S=Southampton<br>
#     &emsp;&emsp;Q=Queenstown<br>
#     &emsp;&emsp;c=Cherbourg<br>
#     </b></p>

# In[114]:


def getcity(passe):
    em=passe['Embarked']
    if em=='Q':
        return 'Queenstown'
    if em=='S':
        return 'Southampton'
    return 'Cherbourg'
titanic_df['City']=titanic_df.apply(getcity,axis=1)
sns.catplot(data=titanic_df,kind='count',x='City',hue='Pclass')


# In[115]:


sns.catplot(data=titanic_df,x="City",hue="Sex",kind='count')


# In[116]:


sns.catplot(data=titanic_df,x='City',hue='Person',kind='count')


# <b>We observe that from Southampton city passengers turned up to be more<b>

# In[117]:


titanic_df.head()


# In[118]:


sns.catplot(x='Embarked',data=deck_df,hue='Person',kind='count')


# In[119]:


sns.catplot(data=deck_df,x="Embarked",hue='Sex',kind='count')


# <h3>Now will try to find relation between survival chances of a passenger and the relatives he/she is travelling with <br>
#     on the same deck</h3>

# <p><b>
#     In the dataset,the SibSp column refers to the number of siblings travelling and Parch refers to parents.<br>
#     If number of SibSp and Parch is equal to zero then the passenger is Alone<br>
#     </b></p>

# In[120]:


def isAlone(human):
    sb=human['SibSp']
    pt=human['Parch']
    return "No" if sb+pt>0 else "Yes"
titanic_df['Alone'] = titanic_df.apply(isAlone,axis=1)


# In[121]:


titanic_df.head()


# In[122]:


sns.catplot(x='Alone',data=titanic_df,hue='Person',kind='count')


# In[123]:


sns.catplot(x='Alone',data=titanic_df,kind='count')


# In[124]:


sns.boxplot(x='Pclass',y='Survived',data=titanic_df)


# In[125]:


sns.catplot(x='Pclass',data=titanic_df,hue='Survived',kind='count')


# <h4>Let's,Find the relation between Survival chances and Pclass</h4>

# In[126]:


def g_survived(p):
    return "Yes" if p['Survived']==1 else "No"
titanic_df['Survived']=titanic_df.apply(g_survived,axis=1)
sns.catplot(x='Pclass',kind='count',hue='Survived',data=titanic_df)


# In[127]:


fig = sns.FacetGrid(titanic_df,hue='Survived',aspect=4)
fig.map(sns.kdeplot,'Pclass',shade=True)
fig.add_legend()


# In[128]:


f_df=titanic_df[(titanic_df.Pclass==1)]
f_df['Pclass'].value_counts()


# In[129]:


s_df=titanic_df[(titanic_df.Pclass==2)]
s_df['Pclass'].value_counts()


# In[130]:


t_df=titanic_df[(titanic_df.Pclass==3)]
t_df['Pclass'].value_counts()


# <b>In Class 1,Survived vs Person.</b>

# In[131]:


class1=sns.catplot(x='Person',data=f_df,hue='Survived',kind='count')
class1.ax.set_title("Class 1")


# <b>In Class 2,Survived vs Person.</b>

# In[132]:


class2=sns.catplot(x='Person',data=s_df,hue='Survived',kind='count')
class2.ax.set_title("Class 2")


# <b>In class 3,Survived vs Person.</b>

# In[133]:


class3=sns.catplot(x='Person',data=t_df,hue='Survived',kind='count')
class3.ax.set_title("Class 3")


# In[134]:


def convst(p):
    return 0 if p['Survived']=='No' else 1
titanic_df['Survivedfs']=titanic_df.apply(convst,axis=1)
sns.lmplot(x='Age',y='Survivedfs',data=titanic_df)


# <b>Let's take a look,if Pclass has any affect on Survival chances distributed among different ages.</b>

# In[135]:


sns.lmplot(x='Age',y='Survivedfs',data=titanic_df)


# <b>We can see from above both graphs that passengers with age greater than 50 has lower chance of surviving.</b>
# <b>The below given plots accurately plot the regression line against Age parameter that swipes in the<br>
#    range from(0.2~0.6)</b>

# In[136]:


sns.regplot(x='Age',y='Survivedfs',data=titanic_df)


# In[137]:


sns.lmplot(x='Age',y='Survivedfs',hue='Pclass',data=titanic_df)


# <h3>Lets's find whether a person has higher survival chance if he is along with his relatives on deck or he isn't.</h3>

# In[138]:


sns.catplot(x='Alone',hue='Survived',stacked=Truedata=titanic_df,kind='count')


# In[139]:


titanic_df['Alone'].value_counts()


# In[140]:


ns=titanic_df['Survived'].value_counts()
ns


# <h4>Now Let's get into math and find the survival probailities of passengers based on different categories.</h4>

# In[141]:


total=titanic_df.shape[0]
num_total_sur=titanic_df[titanic_df['Survived']=="Yes"].shape[0]
alone=titanic_df[titanic_df['Alone']=="Yes"]
not_alone=titanic_df[titanic_df['Alone']=="No"]
num_alone=alone.shape[0]
num_not_alone=not_alone.shape[0]
num_alone_sur=alone[alone['Survived']=="Yes"].shape[0]
num_not_alone_sur=not_alone[not_alone['Survived']=="Yes"].shape[0]
f_sur=f_df[f_df['Survived']=='Yes']
num_first=f_df.shape[0]
num_first_sur=f_sur.shape[0]
s_sur=s_df[s_df['Survived']=='Yes']
num_second=s_df.shape[0]
num_second_sur=s_sur.shape[0]
t_sur=t_df[t_df['Survived']=='Yes']
num_third=t_df.shape[0]
num_third_sur=t_sur.shape[0]
male=titanic_df[titanic_df['Person']=='male']
num_male=male.shape[0]
num_male_sur=male[male['Survived']=="Yes"].shape[0]
female=titanic_df[titanic_df['Person']=='female']
num_female=female.shape[0]
num_female_sur=female[female['Survived']=="Yes"].shape[0]
child=titanic_df[titanic_df['Person']=='child']
num_child=child.shape[0]
num_child_sur=child[child['Survived']=="Yes"].shape[0]
S_em=titanic_df[titanic_df['Embarked']=="S"]
num_s=S_em.shape[0]
num_s_sur=S_em[S_em['Survived']=="Yes"].shape[0]
Q_em=titanic_df[titanic_df['Embarked']=="Q"]
num_q=Q_em.shape[0]
num_q_sur=Q_em[Q_em['Survived']=="Yes"].shape[0]
C_em=titanic_df[titanic_df['Embarked']=="C"]
num_c=C_em.shape[0]
num_c_sur=C_em[C_em['Survived']=="Yes"].shape[0]


# In[142]:


print("------------------Probabilities:------------------------")
print("probability for alone passengers survival::",num_alone_sur/num_alone)
print("probability for family passengers survival::",num_not_alone_sur/num_not_alone)
print("probability for male survival::",num_male_sur/num_male)
print("probability for female survival::",num_female_sur/num_female)
print("probability for children survival::",num_child_sur/num_child)
print("probability for first class passengers survival::",num_first_sur/num_first)
print("probability for second class passengers survival::",num_second_sur/num_second)
print("probability for third class passengers survival::",num_third_sur/num_third)
print("probability for  Southampton passengers survival::",num_s_sur/num_s)
print("probability for  Queenstown passengers survival::",num_q_sur/num_q)
print("probability for  Cherbourg passengers survival::",num_c_sur/num_c)
print()
print()
print("probability for Total passengers survival::",num_total_sur/total)


# <p><b>
#     All in all , breifly to conclude :
#     &emsp;Data Analysis domain with python has versatile opportunities which when gone through<br>
#     &emsp;will provide better and valueble insights on the given data. Hope this breif analysis will<br>
#     &emsp;sigh good interest in my future projects <br>
#     </b></p>

# <h2> Many Yet To Discover ..............................................................By Gokula Chandra .K</h2>
