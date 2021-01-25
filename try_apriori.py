#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt
import random


# In[2]:


from IPython.display import display, HTML


# In[3]:


df = pd.read_csv("../train_ver2.csv")


# In[4]:


df


# In[5]:


for col in df.columns:
    print("@" + col)
    temp = df.loc[:, col].value_counts()
    print(temp)
    print("-"*15)
    print("total unique: " + str(len(temp)))
    print("="*30)


# In[6]:


df.columns 


# In[ ]:





# ### pie chart for all

# In[7]:


colors = []
for i in range(24):
    colors.append((round(random.uniform(0.5, 1), 2), round(random.uniform(0.5, 1), 2), round(random.uniform(0.5, 1), 2)))


# In[8]:


colors


# In[9]:


df_sum = df.iloc[:,24:].sum()
plt.figure(figsize = (20, 20))
plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
plt.axis('equal')
plt.title("pie chart for all", fontsize=24)
plt.savefig("pie_chart_all" + ".jpg")


# In[ ]:





# ### pie chart for different month

# In[10]:


for month in df.loc[:, 'fecha_dato'].unique():
    df_sum = df[df.loc[:, 'fecha_dato'] == month].iloc[:,24:].sum()
    plt.figure(figsize = (20, 20))
    plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title(month, fontsize=24)
    #plt.savefig("pie_chart_" + month + ".jpg")


# ### pie chart for different gender

# In[11]:


for s in ['H', 'V']:
    df_sum = df[df.loc[:, 'sexo'] == s].iloc[:,24:].sum()
    plt.figure(figsize = (20, 20))
    plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title("Gender: " + s, fontsize=24, loc = 'left')
    #plt.savefig("pie_chart_sexo" + s + ".jpg")


# In[ ]:





# In[ ]:





# ### pie chart for different empleado

# In[12]:


for emp in df.loc[:, 'ind_empleado'].unique():
    if pd.isna(emp):
        print("There is a Nan...")
        continue
    df_sum = df[df.loc[:, 'ind_empleado'] == emp].iloc[:,24:].sum()
    plt.figure(figsize = (20, 20))
    plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title("ind_empleado: " + emp, fontsize=24, loc = 'left')
    #plt.savefig("pie_chart_ind_empleado_" + emp + ".jpg")


# In[ ]:





# ### pie chart for different segmento

# In[13]:


for seg in df.loc[:, 'segmento'].unique():
    if pd.isna(seg):
        print("There is a Nan...")
        continue
    df_sum = df[df.loc[:, 'segmento'] == seg].iloc[:,24:].sum()
    plt.figure(figsize = (20, 20))
    plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title("segmento: " + seg, fontsize=24, loc = 'left')
    #plt.savefig("pie_chart_segmento_" + seg + ".jpg")


# In[ ]:





# ### pie chart for pais_residencia

# In[ ]:


for res in df.loc[:, 'pais_residencia'].unique():
    if pd.isna(res):
        print("There is a Nan...")
        continue
    df_sum = df[df.loc[:, 'pais_residencia'] == res].iloc[:,24:].sum()
    plt.figure(figsize = (8, 8))
    plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title("pais_residencia: " + res, fontsize=24, loc = 'left')
    plt.show()
    #plt.savefig("pie_chart_pais_residencia_" + seg + ".jpg")


# ### pie chart for tiprel_1mes

# In[ ]:


for lme in df.loc[:, 'tiprel_1mes'].unique():
    if pd.isna(lme):
        print("There is a Nan...")
        continue
    df_sum = df[df.loc[:, 'tiprel_1mes'] == lme].iloc[:,24:].sum()
    plt.figure(figsize = (20, 20))
    plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title("tiprel_1mes: " + lme, fontsize=24, loc = 'left')
    plt.show()
    #plt.savefig("pie_chart_tiprel_1mes_" + seg + ".jpg")


# ### pie chart for nomprov

# In[ ]:


for nom in df.loc[:, 'nomprov'].unique():
    if pd.isna(nom):
        print("There is a Nan...")
        continue
    df_sum = df[df.loc[:, 'nomprov'] == nom].iloc[:,24:].sum()
    plt.figure(figsize = (10, 10))
    plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title("nomprov: " + nom, fontsize=24, loc = 'left')
    plt.show()
    #plt.savefig("pie_chart_nomprov_" + seg + ".jpg")


# ### pie chart for conyuemp

# In[ ]:


for i in df.loc[:, 'conyuemp'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    df_sum = df[df.loc[:, 'conyuemp'] == i].iloc[:,24:].sum()
    plt.figure(figsize = (10, 10))
    plt.pie(df_sum, labels = df_sum.index, autopct = '%1.2f%%', colors = colors, textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title("conyuemp: " + i, fontsize=24, loc = 'left')
    plt.show()
    #plt.savefig("pie_chart_nomprov_" + seg + ".jpg")


# In[ ]:





# ### rule for all

# In[14]:


df_temp = df.iloc[:,24:]
df_temp = df_temp.fillna(0)
freq = apriori(df_temp, min_support=0.05, use_colnames=True)
result = association_rules(freq, metric="confidence", min_threshold=0.66)
#print("@For all ")
#print(result.sort_values(by = ['confidence'],ascending=False))
result.sort_values(by = ['confidence'],ascending=False).to_csv("all" + "_0.05_0.66.csv")


# In[15]:


temp = pd.read_csv("all" + "_0.05_0.66.csv")
temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
cols = ['antecedents','consequents']
temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
print("For all: ")
pd.set_option('display.max_colwidth', 100)
display(temp.head(20))


# In[ ]:





# ### 分segmento算

# In[16]:


for seg in df.loc[:, 'segmento'].unique():
    if pd.isna(seg):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'segmento'] == seg]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@segmento " + seg)
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("segmento_" + seg + "_0.025_0.75.csv")


# In[17]:


for seg in df.loc[:, 'segmento'].unique():
    if pd.isna(seg):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("segmento_" + seg + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For segmento: " + seg)
    pd.set_option('display.max_colwidth', 100)
    display(temp.head(20))


# ## Association Rules

# ### 分ind_empleado算

# In[18]:


for emp in df.loc[:, 'ind_empleado'].unique():
    if pd.isna(emp):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'ind_empleado'] == emp]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@ind_empleado " + emp)
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("ind_emplead_" + emp + "_0.025_0.75.csv")


# In[19]:


for emp in df.loc[:, 'ind_empleado'].unique():
    if pd.isna(emp):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("ind_emplead_" + emp + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For empleado: " + emp)
    pd.set_option('display.max_colwidth', 150)
    display(temp.head(20))
    


# ### 分月份算（全部顧客的購物清單一起）

# In[21]:


for month in df.loc[:, 'fecha_dato'].unique():
    df_temp = df[df.loc[:, 'fecha_dato'] == month]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.05, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@" + month)
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv(month + "_0.05_0.75.csv")


# In[22]:


for month in df.loc[:, 'fecha_dato'].unique():
    temp = pd.read_csv(month + "_0.05_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For " + month)
    pd.set_option('display.max_colwidth', 100)
    display(temp.head(20))


# In[ ]:





# ### 分顧客性別算（個別顧客每個月的清單）

# In[23]:


for s in ['H', 'V']:
    df_temp = df[df.loc[:, 'sexo'] == s]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@sexo: " + s)
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv( "sexo_" + s + "_0.025_0.75.csv")


# In[24]:


for s in ['H', 'V']:    
    temp = pd.read_csv("sexo_" + s + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For gender: " + s)
    pd.set_option('display.max_colwidth', 100)
    display(temp.head(20))


# In[ ]:





# ### 分ind_actividad_cliente

# In[ ]:


for i in df.loc[:, 'ind_actividad_cliente'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'ind_actividad_cliente'] == i]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@ind_actividad_cliente " + str(i))
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("ind_actividad_cliente_" + str(i) + "_0.025_0.75.csv")


# In[ ]:


for i in df.loc[:, 'ind_actividad_cliente'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("ind_actividad_cliente_" + str(i) + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For ind_actividad_cliente: " + str(i))
    pd.set_option('display.max_colwidth', 150)
    display(temp.head(20))


# In[ ]:





# ### 分indfall

# In[ ]:


for i in df.loc[:, 'indfall'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'indfall'] == i]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@indfall " + str(i))
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("indfall_" + str(i) + "_0.025_0.75.csv")


# In[ ]:


for i in df.loc[:, 'indfall'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("indfall_" + str(i) + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For indfall: " + str(i))
    pd.set_option('display.max_colwidth', 150)
    display(temp.head(20))


# In[ ]:





# ### 分conyuemp 

# In[ ]:


for i in df.loc[:, 'conyuemp'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'conyuemp'] == i]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@conyuemp" + str(i))
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("conyuemp_" + str(i) + "_0.025_0.75.csv")


# In[ ]:


for i in df.loc[:, 'conyuemp'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("conyuemp_" + str(i) + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For conyuemp: " + str(i))
    pd.set_option('display.max_colwidth', 150)
    display(temp.head(20))


# In[ ]:





# ### 分 indext 

# In[ ]:


for i in df.loc[:, 'indext'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'indext'] == i]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@indext" + str(i))
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("indext_" + str(i) + "_0.025_0.75.csv")


# In[ ]:


for i in df.loc[:, 'indext'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("indext_" + str(i) + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For indext: " + str(i))
    pd.set_option('display.max_colwidth', 150)
    display(temp.head(20))


# In[ ]:





# ### 分 indresi

# In[ ]:


for i in df.loc[:, 'indresi'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'indresi'] == i]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@indresi" + str(i))
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("indresi_" + str(i) + "_0.025_0.75.csv")


# In[ ]:


for i in df.loc[:, 'indresi'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("indresi_" + str(i) + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For indresi: " + str(i))
    pd.set_option('display.max_colwidth', 150)
    display(temp.head(20))


# In[ ]:





# ### 分 indrel_1mes 

# In[ ]:


for i in df.loc[:, 'indrel_1mes'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'indrel_1mes'] == i]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@indrel_1mes" + str(i))
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("indrel_1mes_" + str(i) + "_0.025_0.75.csv")


# In[ ]:


for i in df.loc[:, 'indrel_1mes'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("indrel_1mes_" + str(i) + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For indrel_1mes: " + str(i))
    pd.set_option('display.max_colwidth', 150)
    display(temp.head(20))


# In[ ]:





# ### 分indrel

# In[ ]:


for i in df.loc[:, 'indrel'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    df_temp = df[df.loc[:, 'indrel'] == i]
    df_temp = df_temp.iloc[:,24:]
    df_temp = df_temp.fillna(0)
    freq = apriori(df_temp, min_support=0.025, use_colnames=True)
    result = association_rules(freq, metric="confidence", min_threshold=0.75)
    #print("@indrel" + str(i))
    #print(result.sort_values(by = ['confidence'],ascending=False))
    result.sort_values(by = ['confidence'],ascending=False).to_csv("indrel_" + str(i) + "_0.025_0.75.csv")


# In[ ]:


for i in df.loc[:, 'indrel'].unique():
    if pd.isna(i):
        print("There is a Nan...")
        continue
    temp = pd.read_csv("indrel_" + str(i) + "_0.025_0.75.csv")
    temp = temp.drop(columns = ["Unnamed: 0", "antecedent support", "consequent support", "support", "lift", "leverage", "conviction"])
    cols = ['antecedents','consequents']
    temp[cols] = temp[cols].applymap(lambda x: x.replace('frozenset', '').replace('})', '').replace('({', '').replace("'", ""))
    print("For indrel: " + str(i))
    pd.set_option('display.max_colwidth', 150)
    display(temp.head(20))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




