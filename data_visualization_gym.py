# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:58:15 2024

@author: User
"""

# Deciding where to open a gym in Singapore - hypothetical problem
'''
The dataset that we are mainly using is the 'Singapore residents by planning area subzone, agegroup, sex and type of dwelling' available on Singstat.com. For relevancy, our team has decided to extrapolate the relevant sheet referring to the number of residents in each subzone. We also used several data that we obtained through manual tallying in different reliable websites, as there was no data available regarding gym distribution or membership prices available on Singstat or Data.gov. In this report, we will utilize
1. totalpopulationfinal.csv.csv
2. sector2.csv
3. gymdistributionfinal.csv
4. gym visitors dataset.csv
5. gymsandprices5.csv
6. gymdistributionfinalfinal4.csv
'''
# First, we import the modules that we will be using for the data visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf

#Population trends in Singapore (Projected demand), 2015 - 2020

year = ['2015','2016','2017','2018','2019','2020']
file = pd.read_csv('totalpopulationfinal.csv')
data = file.iloc[0].str.replace(',','').astype(int)
data_in_millions = data/1000000

scale_factor = 0.5
plt.figure(figsize=(11,5),dpi=300)
plt.plot(year,data_in_millions, color = 'b', marker='o')
plt.title("Overall Population trend in Singapore",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 25)
plt.ylim(5.5,5.8)
    
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Years',fontsize=16)
plt.ylabel('Total Population (in millions)',fontsize=16)

plt.show()

# Resident Population by Planning Area/Subzone and Type of Dwelling
sector = pd.read_csv('sector2.csv')
sector.dropna(inplace=True)

different_sector = sector['planning_area'].unique()
print(different_sector)

### Factor of Consideration #1: Per Sector Density (2019)
#2019 per sector density
years = sector['year'].unique()
years
subset_2019 = sector.loc[(sector['year'] == 2019) & (sector['resident_count'] > 0)]
subset_2019

subset_2019['planning_area'] == 'Yishun'
groups = subset_2019.groupby(['planning_area']).sum()

residents_per_sector = groups['resident_count']

plt.figure(figsize=(20,10),dpi=300)
plt.bar(residents_per_sector.index, residents_per_sector, color = 'grey')
plt.xticks(residents_per_sector.index, rotation='vertical',fontsize=12)
plt.yticks(fontsize=12)
bar = plt.bar(residents_per_sector.index, residents_per_sector, color = 'grey')
bar[1].set_color('orange')
bar[40].set_color('orange')
bar[41].set_color('orange')
bar[14].set_color('orange')
bar[12].set_color('orange')
bar[36].set_color('orange')
bar[31].set_color('orange')
bar[25].set_color('orange')
bar[8].set_color('orange')

plt.title("Total Number of Residents in Each Sector (2019)",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 30)
plt.ylabel ('Number of residents in each sector',fontsize=16)
plt.xlabel('Different regions in Singapore', fontsize=16)

plt.show()


index = subset_2019
density = residents_per_sector
top9 = density.sort_values(ascending=False).iloc[:9]
regions_names = top9.index
sum_of_regions = top9.sum()

#find the percentage of each region
percentage1 = top9/sum_of_regions
print(percentage1)

plt.figure(figsize=(20,10),dpi=300)
plt.bar(regions_names,percentage1, color = 'lightcoral', width=0.6)
plt.title("Percentage of Residents in Each Region / Total Residents in Selected Regions",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 30)
plt.ylabel ('Percentage of Each Region',fontsize=16)
plt.xlabel('Selected Regions',fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.show()


density = residents_per_sector
density = density.sort_values(ascending=False).iloc[:9]


population = []
area = []
for i in range(0,9):
    population.append(density[i])
    area.append(density.index[i])


# Plot a pie chart for the top 9 areas based on population density
  
labels = area
sizes = population
colors = ['#ffcccc', '#ffe6cc', '#ccffcc','#e6f0ff', '#e6ccff', '#cceeff','#ffffcc', '#e0e0eb', '#f9f2ec']

fig1, ax1= plt.subplots(figsize=(13, 8))
ax1.pie(sizes, labels=labels, colors = colors, startangle=90, pctdistance = 0.7, shadow = True, textprops={'fontsize': 13},  autopct='%1.1f%%',)
ax1.axis('equal')

plt.show()

### Factor of Consideration #2: Number of Gyms in each Region

sector = pd.read_csv('gymdistributionfinal.csv')
sector.dropna(inplace=True)

different_sector = sector['Sub Zone'].unique()
print(different_sector)

data_gym = pd.read_csv("gymdistributionfinal.csv")
sub_zone=data_gym.groupby('Sub Zone')
number_gyms= sub_zone["Gym"].nunique()

plt.figure(figsize=(20,10))
bar1=plt.bar(number_gyms.index,number_gyms, color = 'grey')
plt.xticks(number_gyms.index, rotation="vertical",fontsize=13)
plt.yticks(fontsize=14)
plt.xlabel("Sector", fontsize=16)
plt.ylabel("Number of gyms", fontsize=16)
plt.title("Gym Distribution in Singapore with Respect to Region",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 30)

bar1[2].set_color('orange')
bar1[13].set_color('orange')
bar1[25].set_color('orange')
bar1[29].set_color('orange')
bar1[42].set_color('orange')
bar1[45].set_color('orange')
bar1[51].set_color('orange')
bar1[59].set_color('orange')
bar1[60].set_color('orange')

plt.show()

weighted = percentage1 * 0.4 - percentage2*0.6
sorted_weights = weighted.sort_values(ascending= False)

plt.figure(figsize=(20,10),dpi=300)
plt.bar(sorted_weights.index,sorted_weights, color = 'lightcoral', width=0.6)
plt.title("Final region score amongst selected regions",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 30)
plt.ylabel ('Final region score',fontsize=16)
plt.xlabel('Selected Regions',fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.show()

## 3.0 Research Question 2: What is the target audience of our gym?
### Factor Of Consideration #1: Number Of Gym Members By Age Group

visitors = pd.read_csv('gym visitors dataset.csv')
visitors.head(6) #visualizing the dataset

visitors.shape # there are 5941 respondents to the survey
# compare percentage of people with gym membership vs no gym membership
visitors['Membership'].value_counts(normalize=True)

# 24.04% of all participants have gym membership. 
# it can be inferred that 1/4 of singaporeans have gym membership, so there is market opportunity

# slicing to get the data for members of gyms
is_member = visitors['Membership']=='Yes'
members = visitors[is_member]

plt.figure(figsize=(11,5),dpi=300)
plt.title("Distribution of Age of Gym Members",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 25)
plt.xlabel('Age of gym members', fontsize=16)
plt.xticks(fontsize=14)
sns.set_style("whitegrid")
sns.boxplot(x=members['Age'])

# split members into diff age group
is_16_24 = (members['Age']>=16) & (members['Age']<=24)
is_25_34 = (members['Age']>=25) & (members['Age']<=34)
is_35_44 = (members['Age']>=35) & (members['Age']<=44)
is_45_54 = (members['Age']>=45) & (members['Age']<=54)
is_55_above = members['Age']>=55

number_16_24 = len(members[is_16_24].index)
number_25_34 = len(members[is_25_34].index)
number_35_44 = len(members[is_35_44].index)
number_45_54 = len(members[is_45_54].index)
number_55_above = len(members[is_55_above].index)

age_groups = pd.Series(['16-24','25-34','35-44','45-54','55+'])
numbers = pd.Series([number_16_24, number_25_34, number_35_44, number_45_54, number_55_above])

plt.figure(figsize=(11,5),dpi=300)
bar_age = plt.barh(age_groups, numbers, alpha=0.5,height=0.6,color='grey')
plt.xticks(np.arange(0,650,50),fontsize=14)
plt.yticks(fontsize=14)
bar_age[1].set_color('blue')
bar_age[2].set_color('blue')
bar_age[0].set_color('blue')
plt.title("Number of Gym Members by Age Group",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 25)

for a,b in zip(age_groups,numbers):
    plt.text(b+0.3, a, b, ha='left',va='center', fontsize=12 )

plt.ylabel('Age Groups',fontsize=20)
plt.xlabel('Number of Gym Members', fontsize=20)
plt.show()

# it seems that gym membership is most popular in the age group of 25-34, followed by 35-44 and 16-24.
# therefore, we should use differentiated marketing stratugy to 
# primarily target young working adults and young highschool to university students 

# grouping gym members into different visit frequency
member_fre = members.groupby('Fre').count()
member_fre['Age']

plt.figure(figsize=(11,5),dpi=300)
plt.bar(member_fre['Age'].index, member_fre['Age'],alpha=0.4, align='center')

plt.xticks(fontsize=14)
plt.yticks(np.arange(0,800,100),fontsize=14)
plt.xlabel('Level of Frequency', fontsize = 14)
plt.ylabel('Number of Gym Members', fontsize =14)
for a,b in zip(member_fre['Age'].index, member_fre['Age']):
    plt.text(a, b+1, round(b/1428,2), ha='center', va='bottom', fontsize=14)

plt.title("Visiting Frequency by Gym Members",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 25)

plt.show()

# we can expect most of the gym members to visit several times a week(44%) or several times a month(35%)

# to visualize the proportion of each age group among the different level of visiting frequency

age_fre_1 = members[is_16_24].groupby('Fre').count()
age_fre_2 = members[is_25_34].groupby('Fre').count()
age_fre_3 = members[is_35_44].groupby('Fre').count()
age_fre_4 = members[is_45_54].groupby('Fre').count()
age_fre_5 = members[is_55_above].groupby('Fre').count()

x = age_fre_1['Age'].index 
# x is the frequencies on a scale of 0 to 4
b_age_2 = age_fre_1['Age']
b_age_3 = list(np.add(b_age_2,age_fre_2['Age']))
b_age_4 = list(np.add(b_age_3,age_fre_3['Age']))
b_age_5 = list(np.add(b_age_4,age_fre_4['Age']))

plt.figure(figsize=(11,5),dpi=300)
plt.bar(x, age_fre_1['Age'], label='16-24', color='lightblue')
plt.bar(x, age_fre_2['Age'], bottom=b_age_2, label='25-34', color='gold')
plt.bar(x, age_fre_3['Age'], bottom=b_age_3, label='35-44', color='mediumaquamarine')
plt.bar(x, age_fre_4['Age'], bottom=b_age_4, label='45-54',color='lightseagreen')
plt.bar(x, age_fre_5['Age'], bottom=b_age_5, label='55+', color='teal')

plt.title("Visiting Frequency by Age Groups",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 25)
plt.xlabel('Level of Frequency', fontsize=20)
plt.ylabel('Number of Gym Members', fontsize=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)

# it can be seen that among the most frequent gym goers(frequency level 3-4),
# majority is from age group 25-34, followed by 35-44 and 16-24.
# thus, marketing could mainly target people from the age group 25-34, 
# as they are more likely to visit frequently thus generate more revenue for the gym

gyms_prices = pd.read_csv('gymsandprices5.csv') #Importing data set for gyms and respective membership prices per month
gyms_prices.dropna(inplace=True)
gyms_prices

plt.figure(figsize=(50, 40))
plt.bar(gyms_prices['Gym'], gyms_prices['Price'], color='r', alpha=0.3,
            label='Prices')
plt.xticks(gyms_prices.index, rotation='vertical', fontsize=30)
plt.yticks(fontsize=50)
plt.xlabel('Gyms', fontsize=40)
plt.ylabel('Membership prices ($)', fontsize=40)
plt.show()

plt.figure(figsize=(50, 30))
plt.hist(gyms_prices['Price'], bins = 40, color='r', alpha=0.5,
            label='Prices')
plt.title("Distribution of monthly membership prices",bbox={'facecolor':'k', 'pad':8},color='w',fontsize = 70)
plt.xticks(np.arange(0, 1000, 100),fontsize=40)
plt.yticks(fontsize=40)
plt.xlabel('Gyms Prices ($)', fontsize=50)
plt.ylabel('Frequency', fontsize=50)
plt.show()

#Visualization of prices of gyms

#Using the 'visitors' dataset and 'gyms_prices' to create a sub data set of gyms, prices & member count

number_members = members['Which Gym '].value_counts()
no_members = number_members.to_frame().reset_index()
no_members.columns = ['Gym','Members Count']
no_members #number of members for each gym 

subset_gym = pd.merge(no_members, gyms_prices, on='Gym', how = 'outer')
subset_gyms = subset_gym.dropna()
subset_gyms 

gym_dist = pd.read_csv('gymdistributionfinalfinal4.csv')
gym_distr = gym_dist.dropna()
gym_distr

gym_count = gym_distr.groupby('Gym').count()
gym_count.dropna().head() #Number of gym outlets are there for each gym

new_subset = pd.merge(subset_gyms, gym_count,on='Gym', how = 'outer') 
#forming new subset with no. of members, prices, and total gym count
newer_subset = new_subset.dropna().rename(columns = {'Price_x':'Price', 'Price_y': 'GymsCount', 'Members Count':'Members'})
#renaming columns
gym_dist_subset= newer_subset.drop(columns =['Sub Zone'])

#dropping unnecessary columns
gym_dist_subset

import statsmodels.formula.api as smf

model_mr = smf.ols('Members ~ Price + GymsCount', gym_dist_subset)
result_mr = model_mr.fit()
print(result_mr.summary())

#MLR for members, prices, and number of gyms


