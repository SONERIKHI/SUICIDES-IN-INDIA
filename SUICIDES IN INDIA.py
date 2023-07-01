#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTING NECESSARY LIBRARIES


# In[2]:


import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm #colormap
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

sns.set_style("dark")

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import geopandas as gpd
import mapclassify


# In[3]:


# READING THE DATASET


# In[4]:


suicides = pd.read_csv(r"C:\Users\DELL\Downloads\New folder\suicides\Suicides in India 2001-2012.csv")


# In[5]:


import os
os.getcwd()


# In[6]:


os.chdir(r"C:\Users\DELL\Downloads\New folder\indian states\Igismap")


# In[7]:


os.listdir()


# In[8]:


india = gpd.read_file(r"C:\Users\DELL\Downloads\New folder\indian states\Igismap\Indian_States.shp")


# In[9]:


suicides.head(10)


# In[10]:


print(f"The size of the dataset:\n Rows:{suicides.shape[0]}\tColumns:{suicides.shape[1]}")


# In[11]:


suicides.isnull().sum()


# In[12]:


# Data Pre-processing


# In[13]:


suicides.columns


# In[14]:


suicides['State'].value_counts()


# In[15]:


suicides.info()


# In[16]:


suicides['Type_code'].value_counts()


# In[17]:


sns.barplot(x='Type_code',y='Total',data=suicides,color='orange').set_title('No of suicides by diiferent causes')
plt.xticks(rotation=20)
plt.ylabel('Average number of suicides')
plt.show()


# In[18]:


suicides = suicides.drop(suicides[suicides.Total==0].index)

print("After Total = 0 is dropped",suicides['Type_code'].unique())

suicides = suicides.drop(suicides[(suicides['State'] == 'Total (Uts)') | (suicides['State'] == 'Total (All India)') | 
               (suicides['State'] == 'Total (States)') | (suicides['State'] == 'D & N Haveli') ].index)

print("After Amb. States are dropped",suicides['Type_code'].unique())

suicides = suicides.drop(suicides[(suicides['Age_group'] == '0-100+')].index)

print("After Age Group Amb is dropped",suicides['Type_code'].unique())


# In[19]:


suicides['State'].replace({'A & N Islands':'Andaman & Nicobar Island',
                        'Delhi (Ut)':'NCT of Delhi',
                       }, inplace = True)

india.rename(columns = {'st_nm':'State'}, inplace = True)
india['State'].replace({'Telangana':'Andhra Pradesh'
                       }, inplace = True)


# In[20]:


# 1. YEARWISE OVERVIEW


# In[21]:


trace1 = go.Bar(
    x = suicides['Year'].unique(),
    y = suicides.groupby('Year').sum()['Total']
)

layout = go.Layout()
fig = go.Figure(data = trace1, layout = layout)

fig.update_layout(
    title="Yearwise suicide",
    xaxis_title="Year",
    yaxis_title="Suicide Count",
)
fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))

iplot(fig)


# In[22]:


# 2. Major Geospatial Analysis Overview


# In[23]:


suicides_states = suicides.groupby(['State']).agg({'Total':'sum','Year':'count'})
suicides_states.rename(columns = {'Year':'Cases Type Count'}, inplace = True)


# In[24]:


# 2.1.statewise suicides map


# In[25]:


suicide_map = india.merge(suicides_states, left_on='State', right_on='State')

suicide_map['coords'] = suicide_map['geometry'].apply(lambda x: x.representative_point().coords[:])
suicide_map['coords'] = [coords[0] for coords in suicide_map['coords']]

fig, ax = plt.subplots(figsize=(22, 15))

cmap = 'Reds'

ax = suicide_map.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')
ax.set_facecolor('white')
ax.set_title('Suicide Cases per State')

for idx, row in suicide_map.iterrows():
   ax.text(row.coords[0], row.coords[1], s=row['Total'], 
           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})

norm = matplotlib.colors.Normalize(vmin=suicide_map['Total'].min(), vmax= suicide_map['Total'].max())
n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)

#suicide_map[suicide_map['Total'] > 0].plot(ax=ax, cmap=cmap, markersize=1)

plt.xticks([])
plt.yticks([])
plt.show()


# In[26]:


# 2.2 bar plot statewise


# In[27]:


trace1 = go.Bar(
    x = suicides_states.sort_values(by = ['Total'],ascending = False).index,
    y = suicides_states.sort_values(by = ['Total'],ascending = False)['Total']
)

layout = go.Layout()
fig = go.Figure(data = trace1, layout = layout)

fig.update_layout(
    title="Statewise Suicide Count",
    xaxis_title="State",
    yaxis_title="Suicide Count",
)

iplot(fig)


# In[28]:


# 3. age


# In[29]:


# 3.1 Age-Group Distribution of Suicide Cases


# In[30]:


import plotly.express as px

df = px.data.tips()
fig = px.histogram(suicides, x="Age_group", y='Total',
                   color_discrete_sequence=['indianred','lightblue'],
                   marginal="violin", # or violin, rug
                  )

fig.update_layout(
    title="Suicide Cause",
    xaxis_title="Age Group",
    yaxis_title="Total Suicide Count",
)

fig.show()


# In[31]:


# 3.2 Year-wise distribution of Age Groups


# In[32]:


import plotly.express as px

df = px.data.tips()
fig = px.histogram(suicides, y="Age_group", x='Total',color = 'Year',
                   color_discrete_sequence = px.colors.qualitative.G10,
                   marginal=None, # or violin, rug
                  )

fig.update_layout(
    title="Suicide for Age Group, Year-wise",
    xaxis_title="Total Suicide Count",
    yaxis_title="Age Group",
)
fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))

fig.show()


# In[33]:


# 3.3 Geospatial Analysis (Young vs Elder)


# In[34]:


suicides_agegroup = suicides.groupby(['Age_group','State']).sum()
suicides_agegroup


# In[35]:


# 3.3.1 Young Age Group State-wise Mapping


# In[36]:


each_age_range = suicides_agegroup.shape[0]//suicides['Age_group'].nunique()

suicide_map_younger = india.merge(suicides_agegroup[:each_age_range*2], left_on= 'State', right_on = 'State')
suicide_map_elder = india.merge(suicides_agegroup[each_age_range*2:], left_on = 'State', right_on = 'State')

fig, ax = plt.subplots(figsize=(22, 15))

cmap = 'Blues'

ax = suicide_map_younger.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')
ax.set_facecolor('white')
ax.set_title('Suicide Cases per State (Young)')

suicide_map_younger['coords'] = suicide_map_younger['geometry'].apply(lambda x: x.representative_point().coords[:])
suicide_map_younger['coords'] = [coords[0] for coords in suicide_map_younger['coords']]

for idx, row in suicide_map_younger.iterrows():
   ax.text(row.coords[0], row.coords[1], s=row['Total'], 
           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})


norm = matplotlib.colors.Normalize(vmin=suicide_map_younger['Total'].min(), vmax= suicide_map_younger['Total'].max())
n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)

plt.xticks([])
plt.yticks([])
plt.show()


# In[37]:


# 3.3.2 Elder Age Group State-wise Mapping


# In[38]:


fig, ax = plt.subplots(figsize=(22, 15))

cmap = 'binary'

ax = suicide_map_elder.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')
ax.set_facecolor('white')
ax.set_title('Suicide Cases per State (Elder)')

suicide_map_elder['coords'] = suicide_map_elder['geometry'].apply(lambda x: x.representative_point().coords[:])
suicide_map_elder['coords'] = [coords[0] for coords in suicide_map_elder['coords']]

for idx, row in suicide_map_elder.iterrows():
   ax.text(row.coords[0], row.coords[1], s=row['Total'], 
           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})

norm = matplotlib.colors.Normalize(vmin=suicide_map_elder['Total'].min(), vmax= suicide_map_elder['Total'].max())
n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)

plt.xticks([])
plt.yticks([])
plt.show()


# In[39]:


# GENDER


# In[40]:


# 4.1 Barplot by Gender


# In[41]:


import plotly.express as px

df = px.data.tips()
fig = px.histogram(suicides, x="Gender", y='Total',color = 'Gender',
                   color_discrete_sequence=['indianred','lightblue'],
                   marginal=None, # or violin, rug
                  )

fig.update_layout(
    title="Gender-wise Distinction",
    xaxis_title="Gender",
    yaxis_title="Total count",
)

fig.show()


# In[42]:


suicides_sex = suicides.groupby(['Gender','State']).sum()


# In[43]:


# 4.3 State-wise Suicide Mapping (Female)


# In[44]:


suicide_map_female = india.merge(suicides_sex[:suicides_sex.shape[0]//2], left_on= 'State', right_on = 'State')
suicide_map_male = india.merge(suicides_sex[suicides_sex.shape[0]//2:], left_on = 'State', right_on = 'State')

fig, ax = plt.subplots(figsize=(22, 15))

cmap = 'PuRd'

ax = suicide_map_female.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')
ax.set_facecolor('white')
ax.set_title('Suicide Cases per State (Female)')

suicide_map_female['coords'] = suicide_map_female['geometry'].apply(lambda x: x.representative_point().coords[:])
suicide_map_female['coords'] = [coords[0] for coords in suicide_map_female['coords']]

for idx, row in suicide_map_female.iterrows():
   ax.text(row.coords[0], row.coords[1], s=row['Total'], 
           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})

norm = matplotlib.colors.Normalize(vmin=suicide_map_female['Total'].min(), vmax= suicide_map_female['Total'].max())
n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)

plt.xticks([])
plt.yticks([])
plt.show()


# In[45]:


# 4.4 State-wise Suicide Mapping (Male)


# In[46]:


fig, ax = plt.subplots(figsize=(22, 15))

cmap = 'Blues'

ax = suicide_map_male.plot(ax=ax, cmap=cmap,column = 'Total',scheme = 'equal_interval',edgecolor = 'black')
ax.set_facecolor('white')
ax.set_title('Suicide Cases per State (Male)')

suicide_map_male['coords'] = suicide_map_male['geometry'].apply(lambda x: x.representative_point().coords[:])
suicide_map_male['coords'] = [coords[0] for coords in suicide_map_male['coords']]

for idx, row in suicide_map_male.iterrows():
   ax.text(row.coords[0], row.coords[1], s=row['Total'], 
           horizontalalignment='center', bbox={'facecolor': 'white', 'alpha':0.8, 'pad': 2, 'edgecolor':'none'})

norm = matplotlib.colors.Normalize(vmin=suicide_map_male['Total'].min(), vmax= suicide_map_male['Total'].max())
n_cmap = cm.ScalarMappable(cmap= cmap, norm = norm)
n_cmap.set_array([])
ax.get_figure().colorbar(n_cmap)

plt.xticks([])
plt.yticks([])
plt.show()


# In[47]:


# 5. Occupation


# In[48]:


# 5.1 Major Breakdown - Generic Causes, Adoption and Professional Reason


# In[49]:


suicides_type = suicides.groupby(['Type_code','Year']).sum()
dist_df = suicides_type.reset_index(level=[0,1]) #Multi-index to Single-index

trace0 = go.Box(y=dist_df.loc[dist_df['Type_code']=='Causes']['Total'],name="Causes")
trace1 = go.Box(y=dist_df.loc[dist_df['Type_code']=='Means_adopted']['Total'],name="Adoption")
trace2 = go.Box(y=dist_df.loc[dist_df['Type_code']=='Professional_Profile']['Total'],name="Professional")
data = [trace0,trace1,trace2]

#fig = px.box(dist_df, y='Year', x="Total", points="all", color = 'Type_code')
iplot(data)


# In[50]:


# 5.2 By Minor Labels - Indepth Causes (Minor Types)


# In[51]:


# 5.2.1 All Minor Labels


# In[52]:


suicides['Type'].unique()


# In[53]:


suicides['Type'].replace({'Others (Please Specify)':'Unspecified',
                      'Causes Not Known':'Unspecified',
                       'Causes Not known':'Unspecified',
                       'Other Causes (Please Specity)':'Unspecified',
                       'By Other means (please specify)':'Unspecified',
                       'By Other means' : 'Unspecified',
                       'Unemployment':'Unemployed'
                      },inplace = True)


# In[54]:


fig = px.histogram(suicides, x="Total", y="Type", color = 'Gender',
                   marginal="violin", # or violin, rug,
                   color_discrete_sequence=['indianred','lightblue'],
                   )

fig.update_layout(
    title="Suicide Cause",
    xaxis_title="Total Suicide Count",
    yaxis_title="Suicide Cause/Method",
)
fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))

fig.show()


# In[55]:


# 5.2.2 Crosschecking Numbers - Unspecified count


# In[56]:


#From Above chart, Unspecified stands at 759k + 307k = 1066k cases of suicide approx

suicides.loc[suicides['Type']=='Unspecified']['Total'].sum()/1000


# In[57]:


# 5.2.3 Top 20 Minor Labels (for cumulative Suicide count of both Genders)


# In[58]:


temp_df = suicides.groupby('Type').sum()['Total']
temp_df = temp_df.reset_index()
temp_df = temp_df.sort_values(by = ['Total'],ascending = False)
temp_df = temp_df[:20]
temp_df.rename(columns = {'Type':'Type Minor',
                          'Total':'Total Cases'},inplace = True)


# In[59]:


temp_df = temp_df.reset_index()
temp_df.drop(columns = ['index'],inplace = True)


# In[60]:


hist_dict = dict(zip(temp_df['Type Minor'], temp_df['Total Cases']))
hist_dict.keys()


# In[61]:


to_select = ['Unspecified', 'By Hanging', 'Family Problems', 'House Wife', 'By Consuming Insecticides', 'By Consuming Other Poison',
             'Farming/Agriculture Activity', 'Other Prolonged Illness', 'Unemployed', 'By Fire/Self Immolation', 'Service (Private)', 
             'By Drowning', 'Insanity/Mental Illness', 'Self-employed (Business activity)', 'Student', 'By coming under running vehicles/trains',
             'Love Affairs', 'Professional Activity', 'Bankruptcy or Sudden change in Economic', 'Poverty']

super_temp = suicides.loc[suicides['Type'].isin(to_select)]


# In[62]:


i = 10
for key in hist_dict:
    
    hist_dict[key] = str(i)+" "+key
    i +=1


# In[63]:


super_temp['Type'].replace(hist_dict, inplace = True)


# In[64]:


super_temp = super_temp.drop(super_temp[(super_temp['Type'] == '10 Unspecified')].index)

fig = px.histogram(super_temp, x="Total", y="Type", color = 'Gender',
                   marginal='violin', # or violin, rug,
                   color_discrete_sequence=['lightblue','indianred'],
                   ).update_yaxes(categoryorder = 'category descending')

fig.update_layout(
    title="Suicide Cause",
    xaxis_title="Total Suicide Count",
    yaxis_title="Suicide Cause/Method",
)
fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))

fig.show()


# In[65]:


# 6. Comparisions


# In[66]:


# 6.1 Age Groups vs Year vs Gender vs Suicide Count


# In[67]:


fig = px.scatter_3d(suicides,x="Year",y="Age_group",z="Total",
    color = 'Gender', size_max = 18,
    color_discrete_sequence=['indianred','lightblue']
    )

fig.show()


# In[68]:


# 6.2 Age and Gender vs Suicide Count


# In[69]:


import plotly.express as px

fig = px.histogram(suicides, x="Age_group", y="Total", color="Gender",
                   color_discrete_sequence=['indianred','lightblue'],
                   marginal="violin", # or violin, rug
                  )

fig.update_layout(
    xaxis_title="Age Group",
    yaxis_title="Suicide count",
)
fig.update_yaxes(tickangle=-30, tickfont=dict(size=7.5))

fig.show()


# In[70]:


# 6.3 Suicide distribution by Major Occupation/Causes Labels


# In[71]:


super_temp2 = suicides.groupby(['Type_code','Gender']).sum()
super_temp2 = super_temp2.reset_index()
super_temp2 = super_temp2.sort_values(by = ['Gender'])

girl_occupation_suicide = super_temp2.iloc[:3]
boy_occupation_suicide = super_temp2.iloc[3:]


# In[72]:


import plotly.graph_objects as go

occupations_major = ['Causes','Means_adopted','Professional Profile']
female_death = list(super_temp2['Total'][:3])
male_death = list(super_temp2['Total'][3:])

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=male_death,
      theta=occupations_major,
      fill='toself',
      name='Male',
    line_color ="lightblue"
))

fig.add_trace(go.Scatterpolar(
      r=female_death,
      theta=occupations_major,
      fill='toself',
      name='Female',
    line_color ="indianred"
))

#fig = px.line_polar(boy_occupation_suicide, r='Total', theta='Type_code', line_close=True)
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0,1000000]
    )),
  showlegend=True
)

fig.update_traces(fill='toself')
fig.show()


# In[73]:


#6.4 Suicide distribution by Major Occupation/Causes Labels


# In[74]:


indexes = suicides.groupby('Type').sum()
indexes.sort_values(by = ['Total'], ascending = False)[1:11].index


# In[75]:


super_temp3 = suicides.groupby(['Gender','Type']).sum()
super_temp3 = super_temp3.reset_index()


# In[76]:


to_choose = ['By Hanging', 'Family Problems', 'House Wife',
       'By Consuming Insecticides', 'By Consuming Other Poison',
       'Farming/Agriculture Activity', 'Other Prolonged Illness', 'Unemployed',
       'By Fire/Self Immolation', 'Service (Private)']

super_temp3_female = super_temp3.loc[ (super_temp3['Type'].isin(to_choose)) & (super_temp3['Gender']=='Female')]
super_temp3_male = super_temp3.loc[ (super_temp3['Type'].isin(to_choose)) & (super_temp3['Gender']=='Male')]

super_temp3_male = super_temp3_male.append({'Type':'House Wife',
                        'Gender':'Male',
                        'Year':0,
                        'Total':0},ignore_index = True)

super_temp3_male = super_temp3_male.sort_values(by = ['Type'])
super_temp3_female = super_temp3_female.sort_values(by = ['Type'])


# In[77]:


# Cummulative Suicide Placard


# In[78]:


col = ['Total Suicide\n Count', 'Most Common\n Cause', 'Most affected \nState', 'Most affected\n Age Group',]

values = [suicides.groupby('Gender').sum()['Total']['Female'] + suicides.groupby('Gender').sum()['Total']['Male'],'By Hanging','Maharashtra','15-29']
color_val = ['lightblue','lightblue','lightblue','lightblue']

fig, axes = plt.subplots(1, 4, figsize=(24, 4))
axes = axes.flatten()
fig.set_facecolor('white')

for ind, col in enumerate(col):
    axes[ind].text(0.5, 0.6, col, 
            ha='center', va='center',
            fontfamily='monospace', fontsize=32,
            color='white', backgroundcolor='#8A0303')

    axes[ind].text(0.5, 0.2, values[ind], 
            ha='center', va='center',
            fontfamily='monospace', fontsize=38, fontweight='bold',
            color='#660000', backgroundcolor='white')
    
    axes[ind].set_axis_off()


# In[79]:


# Male Suicide Placard


# In[80]:


col = ['Suicide Count', 'Most Common\n Cause', 'Most affected \nState', 'Most affected\n Age Group',]

values = [suicides.groupby('Gender').sum()['Total']['Male'],'By Hanging','Maharashtra','30-44']

fig, axes = plt.subplots(1, 4, figsize=(24, 4))
axes = axes.flatten()
fig.set_facecolor('white')

for ind, col in enumerate(col):
    axes[ind].text(0.5, 0.6, col, 
            ha='center', va='center',
            fontfamily='monospace', fontsize=32,
            color='white', backgroundcolor='#2D383A')

    axes[ind].text(0.5, 0.2, values[ind], 
            ha='center', va='center',
            fontfamily='monospace', fontsize=38, fontweight='bold',
            color='#2887C8', backgroundcolor='white')
    
    axes[ind].set_axis_off()


# In[81]:


# Female Suicide Placard


# In[82]:


col = ['Suicide Count', 'Most Common\n Cause', 'Most affected \nState', 'Most affected\n Age Group',]

values = [suicides.groupby('Gender').sum()['Total']['Female'],'Being Housewife','West\n Bengal','15-29']

fig, axes = plt.subplots(1, 4, figsize=(24, 4))
axes = axes.flatten()
fig.set_facecolor('white')

for ind, col in enumerate(col):
    axes[ind].text(0.5, 0.6, col, 
            ha='center', va='center',
            fontfamily='monospace', fontsize=32,
            color='white', backgroundcolor='#2D383A')

    axes[ind].text(0.5, 0.2, values[ind], 
            ha='center', va='center',
            fontfamily='monospace', fontsize=38, fontweight='bold',
            color='indianred', backgroundcolor='white')
    
    axes[ind].set_axis_off()

