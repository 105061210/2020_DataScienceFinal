#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import gc
import xgboost
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


TRAIN_PATH = "train_ver2.csv"


# In[3]:


months = ['2015-01-28', '2015-02-28', '2015-03-28', '2015-04-28', '2015-05-28',
         '2015-06-28', '2015-07-28', '2015-08-28', '2015-09-28', '2015-10-28',
         '2015-11-28', '2015-12-28', '2016-01-28', '2016-02-28', '2016-03-28',
         '2016-04-28', '2016-05-28']


# In[4]:


prods = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 
        'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 
        'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 
        'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 
        'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
        'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
        'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
        'ind_nomina_ult1', 'ind_nom_pens_ult1',  'ind_recibo_ult1']

# exclude 'ind_aval_fin_ult1', 'ind_ahor_fin_ult1'
targetprods = ['ind_cco_fin_ult1', 'ind_recibo_ult1', 'ind_nom_pens_ult1',
    'ind_nomina_ult1', 'ind_tjcr_fin_ult1', 'ind_reca_fin_ult1', 
    'ind_cno_fin_ult1', 'ind_ecue_fin_ult1', 'ind_dela_fin_ult1',
    'ind_deco_fin_ult1', 'ind_ctma_fin_ult1', 'ind_fond_fin_ult1',
    'ind_ctop_fin_ult1', 'ind_valo_fin_ult1', 'ind_ctpp_fin_ult1',
    'ind_ctju_fin_ult1', 'ind_deme_fin_ult1', 'ind_plan_fin_ult1',
    'ind_cder_fin_ult1', 'ind_pres_fin_ult1', 'ind_hip_fin_ult1',
    'ind_viv_fin_ult1']

product_dict = dict(zip(range(len(targetprods)), targetprods))


# In[5]:


targetprods = ['ind_recibo_ult1', 'ind_cco_fin_ult1', 'ind_nom_pens_ult1',
    'ind_nomina_ult1', 'ind_tjcr_fin_ult1', 'ind_ecue_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctma_fin_ult1', 'ind_reca_fin_ult1',
    'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_valo_fin_ult1']


# In[6]:


# total 24 features
features = ['fecha_dato', 'ncodpers', 'ind_empleado', 
       'pais_residencia', 'sexo', 'age', 'fecha_alta', 'ind_nuevo', 
       'antiguedad', 'indrel', 'ult_fec_cli_1t', 'indrel_1mes',
       'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada',
       'indfall', 'tipodom', 'cod_prov', 'nomprov',
       'ind_actividad_cliente', 'renta', 'segmento']


# In[7]:


def get_monthly_data(datos, month_list, feature_list, product_list):
    index = datos.index[datos.loc[:, "fecha_dato"].isin(month_list)]
    valid_cols = feature_list + product_list
    monthly_data = pd.read_csv(TRAIN_PATH, usecols = valid_cols, skiprows = range(1, index[0]+1), nrows = len(index), header = 0)
    return monthly_data 


# In[8]:


def process_features(cur_data, pre_data):
    print("@@@ For sexo: ")
    cur_data.loc[:, "sexo"] = cur_data.loc[:, "sexo"].map(lambda x: 0 if x=='H' else 1).astype(int)
    print(cur_data.loc[:, "sexo"].value_counts())
    
    print("@@@ For ind_empleado: ")
    cur_data.loc[:, "ind_empleado"] = cur_data.loc[:, "ind_empleado"].map(lambda x: 1 if x=='S'                                                                          else 2 if x=='A'                                                                          else 3 if x=='B'                                                                          else 4 if x=='F'                                                                          else 0)
    
    print("@@@ For age: ")
    if cur_data.loc[:, "age"].dtype != np.int64 and cur_data.loc[:, "age"].dtype != np.float64:
        cur_data.loc[:, "age"] = cur_data.loc[:, "age"].str.strip()
        cur_data.loc[:, "age"] = cur_data.loc[:, "age"].map(lambda x: None if x=="NA" else int(x))
    cur_data.loc[:, "age"].fillna(cur_data.loc[:, "age"].median(), inplace=True)
    print(cur_data.loc[:, "age"].describe())
    
    print("@@@ For age categorical: ")
    cur_data["agecateg"] = cur_data.loc[:, "age"].map(lambda x:                                                      0 if x<18                                                      else 1 if (x>=18 and x<25)                                                      else 2 if (x>=25 and x<35)                                                      else 3 if (x>=35 and x<45)                                                      else 4 if (x>=45 and x<55)                                                      else 5 if (x>=55 and x<65)                                                      else 6 if (x>=65) else 7).astype(int)
    print(cur_data.loc[:, "agecateg"].value_counts())
    
    print("@@@ For new customer index: ")
    cur_data.loc[:, "ind_nuevo"].fillna(1.0, inplace=True)
    
    print("@@@ For antiguedad: ")
    if cur_data.loc[:, "antiguedad"].dtype != np.int64 and cur_data.loc[:, "antiguedad"].dtype != np.float64:
        cur_data.loc[:, "antiguedad"] = cur_data.loc[:, "antiguedad"].str.strip()
        cur_data.loc[:, "antiguedad"] = cur_data.loc[:, "antiguedad"].map(lambda x: None if x=='NA' else int(x))
        cur_data.loc[:, "antiguedad"][cur_data.loc[:, "antiguedad"]<0] = cur_data.loc[:, "antiguedad"].max()
        cur_data.loc[:, "antiguedad"].fillna(cur_data.loc[:, "antiguedad"].median(), inplace=True)
        print(cur_data.loc[:, "antiguedad"].describe())
        
    print("@@@ For indrel: ")
    cur_data.loc[:, "indrel"].fillna(1.0, inplace=True)
    
    print("@@@ For indrel_1mes /customer type: ")
    cur_data.loc[:, "indrel_1mes"].fillna(0, inplace=True)
    if cur_data.loc[:, "indrel_1mes"].dtype != np.int64 and cur_data.loc[:, "indrel_1mes"].dtype != np.float64:
        cur_data.loc[:, "indrel_1mes"] = cur_data.loc[:, "indrel_1mes"].str.strip()
        cur_data.loc[:, "indrel_1mes"] = cur_data.loc[:, "indrel_1mes"].map(lambda x:                                                                            0 if x=='NA'                                                                            else 5 if x=='P'                                                                            else (float(x)))
    cur_data.loc[:, "indrel_1mes"].fillna(0, inplace=True)
    print(cur_data.loc[:, "indrel_1mes"].value_counts())
    
    print("@@@ For tiprel_1mes /customer relation type: ")
    cur_data.loc[:, "tiprel_1mes"].fillna('I', inplace=True)
    cur_data.loc[:, "tiprel_1mes"] = cur_data.loc[:, "tiprel_1mes"].map(lambda x:                                                                        0 if x=='I'                                                                        else 1 if x=='A'                                                                        else 2 if x=='P'                                                                        else 3 if x=='R'                                                                        else 4).astype(int)
    print(cur_data.loc[:, "tiprel_1mes"].value_counts())
    
    print("@@@ For indresi: ")
    cur_data.loc[:, "indresi"] = cur_data.loc[:, "indresi"].map(lambda x: 1 if x=='S' else 0).astype(int)
    
    print("@@@ For indext")
    cur_data.loc[:, "indext"] = cur_data.loc[:, "indext"].map(lambda x: 1 if x=='S' else 0).astype(int)
    
    print("@@@ For conyuemp")
    cur_data.loc[:, "conyuemp"] = cur_data.loc[:, "conyuemp"].map(lambda x: 1 if x=='S' else 0).astype(int)
    
    print("@@@ For deceased client /indfall: ")
    cur_data.loc[:, "indfall"] = cur_data.loc[:, "indfall"].map(lambda x: 1 if x=='S' else 0).astype(int)
    
    print("@@@ For province code: ")
    cur_data.loc[:, "cod_prov"].fillna(99, inplace=True)
    
    print("@@@ For ind_actividad_cliente:")
    cur_data.loc[:, "ind_actividad_cliente"].fillna(0, inplace=True)
    
    print("@@@ For segmento: ")
    cur_data.loc[:, "segmento"] = cur_data.loc[:, "segmento"].map(lambda x:                                                                  1 if x=='01 - TOP'                                                                  else 3 if x=='03 - UNIVERSITARIO'                                                                  else 2).astype(int)
    
    print("@@@ For income /renta: ")
    cur_data.loc[:, "renta"] = pd.to_numeric(cur_data.loc[:, "renta"], errors='coerce')
    print("Fill missing income with medians...")
    for ac in cur_data.loc[:, "agecateg"].unique():
        for seg in cur_data.loc[:, "segmento"].unique():
            med = cur_data[(cur_data.loc[:, "agecateg"] == ac) &                            (cur_data.loc[:, "segmento"] == seg)]['renta'].dropna().median()
            cur_data.loc[(cur_data.loc[:, "renta"].isnull()) & (cur_data.loc[:, "agecateg"]==ac)                          &(cur_data.loc[:, "segmento"]==seg)] = med
            
    Xdata = pd.DataFrame(cur_data.loc[:, ['ncodpers', 'sexo', 'age', 'agecateg',
                                      'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes', 'tiprel_1mes',
                                       'indresi', 'indext', 'conyuemp', 'indfall', 'cod_prov', 'ind_actividad_cliente', 'segmento', 
                                       'renta']])
    print("### Col for Xdata: ", Xdata.columns)
    del cur_data
    gc.collect()
    
    print("Merge with previous 'products'")
    X = pd.merge(Xdata, pre_data, how='left', on='ncodpers')
    print("Shape after process and merge: ", X.shape)
    print(X.info())
    X.fillna(0, inplace=True)
    
    return X
    
    


# In[9]:


def added_products(cur_data, pre_data):
    intsec = np.intersect1d(cur_data.loc[:, "ncodpers"], pre_data.loc[:, "ncodpers"])
    print("Size of intsec: ", intsec.size)
    print("Unique size: ", np.unique(intsec).size)
    print("\n Merge...")
    
    mgd = pd.merge(cur_data, pre_data, how='left', on='ncodpers')
    print("Shape of mgd: ", mgd.shape)
    mgd.fillna(0, inplace=True)
    added = pd.DataFrame(mgd.loc[:, "ncodpers"])
    print(added.head())
    
    ### find the difference between this month and previous month
    for i, product in enumerate(targetprods):
        added[product] = mgd.loc[:, product+'_x'] - mgd.loc[:, product+'_y']
        added.loc[added[product] == -1, product] = 0
        
    print(added.head())
    print("### Total added products: ")
    print(added.sum(axis=0))
    
    return added.drop(['ncodpers'], axis=1)


# In[10]:


def lagged_features(dates, mgd, month_list):
    for month in month_list:
        print("@ ", month)
        lag_data = get_monthly_data(dates, [month], ['ncodpers'], prods)
        print(lag_data.shape)
        
        print('\n Merge lagged month ' + str(month) + "...")
        i = month_list.index(month)
        
        mgd = pd.merge(mgd, lag_data, how='left', on='ncodpers', suffixes=[i, i+1])
        print("Shape of mgd: ", mgd.shape)
        #print(mgd.info())
        mgd.fillna(0, inplace=True)
        print(mgd.info())
        
    print("After merging...")
    print(mgd.info())
    
    return mgd


# In[11]:


def apk(actual, predicted, k=7):
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


# In[12]:


def mean_apk(actual, predicted, k=7):
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


# In[22]:


def training(thismonth = '2015-05-28'):
    print('\nLoading dates...')
    datos = pd.read_csv(TRAIN_PATH, usecols=['fecha_dato'], header=0)
    prevmonth = months[months.index(thismonth) - 1]
    print('\nThis month: %s. Previous month: %s' % (thismonth, prevmonth))
    curdata = get_monthly_data(datos, [thismonth], features, prods)
    predata = get_monthly_data(datos, [prevmonth], ["ncodpers"], prods)
    
    print('Get train data (this month client features + prev month prods')
    X = process_features(curdata[features], predata)
    print("Shape of X after processing features: ", X.shape)
    
    print("Lagged features")
    lag1month = months[months.index(prevmonth) - 1]
    lag2month = months[months.index(lag1month) - 1]
    lag3month = months[months.index(lag2month) - 1]
    lag4month = months[months.index(lag3month) - 1]
    lag5month = months[months.index(lag4month) - 1]
    lag6month = months[months.index(lag5month) - 1]
    lag7month = months[months.index(lag6month) - 1]
    lagmonths = [lag1month, lag2month, lag3month, lag4month]
    print('Lagged months: ' + str(lagmonths))
    X = lagged_features(datos, X, lagmonths)
    print("Shape of X after lagged: ", X.shape)
    
    X.drop(['ncodpers'], axis=1, inplace=True)
    
    print('\nAdded products (targets)')
    y = added_products(curdata[['ncodpers']+prods], predata)
    print("Shape of y after added: ", y.shape)
    print(y.values.sum()/y.size)
    print(y[:5])
    
    del curdata, predata
    gc.collect()
    
    print('Training and validation sets')
    Xtrain, Xval, ytrain, yval = train_test_split(X, y, test_size=0.2,random_state=0)
    print(Xtrain.shape, ytrain.shape, Xval.shape, yval.shape)
    
    
    del X, y
    gc.collect()
    
    print('Select only clients with added products')
    addedprods= np.sum(ytrain, axis=1)
    Xtrain = Xtrain[addedprods!=0]
    ytrain = ytrain[addedprods!=0]
    print(Xtrain.shape, ytrain.shape)
    print(ytrain[:5])
    
    targlist=[]
    for row in yval.values:
        clientlist = []
        for i in range(yval.shape[1]):
            if row[i] == 1:
                clientlist.append(product_dict[i])
        targlist.append(clientlist)
        
    print('\n$$$ Training...')
    clfdict = {}
    probs = []
    freq = ytrain.sum(axis=0)
    
    for pr in targetprods:
        print("@@@", pr)
        clf = xgboost.XGBClassifier(max_depth=6, learning_rate = 0.05, 
                subsample = 0.9, colsample_bytree = 0.9, n_estimators=100,
                base_score = freq[pr]/Xtrain.shape[0], nthread=2)
        clfdict[pr] = clf
        clf.fit(Xtrain, ytrain.loc[:, pr])
        ypredv = clf.predict(Xval)
        res = classification_report(yval.loc[:, pr], ypredv)
        print(res)
        probs.append(clf.predict_proba(Xval)[:, 1])
    
    probs = np.array(probs).T
    print(probs.shape) # m n
    
    idsort7 = np.argsort(probs, axis=1)[:, :-8:-1] # ids of seven greatest probs
    prlist = [[product_dict[j] for j in irow] for irow in idsort7]
    mapscore = mean_apk(targlist, prlist, 7)
    print('MAP@7 score: %0.5f' % mapscore)
    
    return clfdict


# In[27]:


def testing(thismonth='2016-05-28', clfdict = {}):
    datos = pd.read_csv(TRAIN_PATH, usecols=['fecha_dato'], header=0)
    prevmonth = months[months.index(thismonth) - 1]
    print('\nTest month: %s. Previous test month: %s' % (thismonth, prevmonth))
    curdata = get_monthly_data(datos, [thismonth], features, prods)
    predata = get_monthly_data(datos, [prevmonth], ["ncodpers"], prods)
    print("curdata col: ", curdata.columns)
    print("predata col: ", predata.columns)
    Xtest = process_features(curdata[features], predata)
    print("Xtest col: ", Xtest.columns)
    print('Lagged test months')
    lag1month = months[months.index(prevmonth) - 1]
    lag2month = months[months.index(lag1month) - 1]
    lag3month = months[months.index(lag2month) - 1]
    lag4month = months[months.index(lag3month) - 1]
    lag5month = months[months.index(lag4month) - 1]
    lagtestmonths= [lag1month, lag2month, lag3month, lag4month]
    print(lagtestmonths)
    Xtest = lagged_features(datos, Xtest, lagtestmonths)
    print("Shape of Xtest after lagged: ", Xtest.shape)
    tids = Xtest['ncodpers']
    Xtest.drop(['ncodpers'], axis=1, inplace=True)
    
    print('\nAdded products (targets)')
    y = added_products(curdata[['ncodpers']+prods], predata)
    print("Shape of y after added: ", y.shape)
    print(y.values.sum()/y.size)
    print(y[:5])
    
    targlist=[]
    for row in y.values:
        clientlist = []
        for i in range(y.shape[1]):
            if row[i] == 1:
                clientlist.append(product_dict[i])
        targlist.append(clientlist)
    
    del curdata
    del predata
    gc.collect()
    tclfdict = clfdict
    testprobs = []
    for pr in targetprods:
        print("@@@@@ ", pr)
        #ypredv = tclfdict[pr].predict(Xtest)
        ypredv = tclfdict[pr].predict_proba(Xtest)[:, 1]
        func = np.vectorize(lambda x: 1 if x>0.6 else 0)
        ypredv = func(ypredv)
        res = classification_report(y.loc[:, pr], ypredv)
        print(res)
        print()
        testprobs.append(tclfdict[pr].predict_proba(Xtest)[:, 1])
        
    testprobs = np.array(testprobs).T
    print("Shape of testprobs: ", testprobs.shape)
    
    
    
    print('Creating list of most probable products...')
    
    idsort5 = np.argsort(testprobs, axis=1)[:, :-6:-1] # ids of seven greatest probs
    predlist5 = [[product_dict[j] for j in irow] for irow in idsort5]
    
    idsort7 = np.argsort(testprobs, axis=1)[:, :-8:-1] # ids of seven greatest probs
    predlist7 = [[product_dict[j] for j in irow] for irow in idsort7]
    
    idsort10 = np.argsort(testprobs, axis=1)[:, :-11:-1] # ids of seven greatest probs
    predlist10 = [[product_dict[j] for j in irow] for irow in idsort10]
    
    mapscore = mean_apk(targlist, predlist5, 5)
    print('MAP@5 score: %0.5f' % mapscore)
    
    mapscore = mean_apk(targlist, predlist7, 7)
    print('MAP@7 score: %0.5f' % mapscore)
    
    mapscore = mean_apk(targlist, predlist10, 10)
    print('MAP@10 score: %0.5f' % mapscore)
    
    return targlist, predlist5, predlist7, predlist10
    


# In[ ]:





# In[15]:


clfdict = training(thismonth = '2016-04-28')


# In[16]:


targlist, predlist5, predlist7, predlist10 = testing(thismonth='2016-05-28', clfdict = clfdict)


# In[17]:


def top1_accuracy(target, pred):
    
    cnt = 0
    emp = 0
    for i in range(len(pred)):
        
        if not target[i]:
            emp +=1
        else:
            #print("-"*15)
            #print("target: ", target[i])
            #print("predic: ", pred[i])
                
            if pred[i][0] in target[i]:
                cnt += 1
                flag = 1
            #print("cnt:    ", cnt)
    print("Total: ", len(pred))
    print("cnt: ", cnt)
    print("emp: ", emp)
    return cnt/(len(pred)-emp)


# In[18]:


top1_accuracy(targlist, predlist5)


# In[36]:


def top5_accuracy(target, pred):
    
    cnt = 0
    emp = 0
    for i in range(len(pred)):
        
        if not target[i]:
            emp +=1
        else:
            flag = 0
            #print("-"*15)
            #print("target: ", target[i])
            #print("predic: ", pred[i])
            for j in range(len(pred[i])):
                
                if pred[i][j] in target[i] and flag==0:
                    cnt += 1
                    flag = 1
            #print("cnt:    ", cnt)
    print("測試筆數:       ", len(pred))
    print("命中:          ", cnt)
    print("target empty: ", emp)
    res = cnt/(len(pred)-emp)
    print("機率:          ", res)
    return res
        
    


# In[39]:


top5_accuracy(targlist, predlist5)


# In[40]:


predlist = predlist5
for i in range(len(predlist)):
    if 'ind_nomina_ult1' in predlist[i]:
        predlist[i] = np.append('ind_nom_pens_ult1', predlist[i])
        predlist[i] = np.append('ind_cno_fin_ult1', predlist[i])
top5_accuracy(targlist, predlist)


# In[21]:


predlist = predlist7

r0a = 0
r1a = 0
r2a = 0
r3a = 0
r4a = 0
r0c = 0
r1c = 0
r2c = 0
r3c = 0
r4c = 0

for i in range(len(predlist)):
    #print("Test " + str(i) + "\t" + str(predlist[i]))
    ### For rule 0
    if 'ind_nomina_ult1' in predlist[i]:
        r0a +=1
        if 'ind_nom_pens_ult1' in predlist[i]:
            r0c += 1
    
    ### For rule 1
    if 'ind_nomina_ult1' in predlist[i] and 'ind_cno_fin_ult1' in predlist[i]:
        r1a +=1
        if 'ind_nom_pens_ult1' in predlist[i]:
            r1c += 1
            
    ### For rule 2
    if 'ind_nomina_ult1' in predlist[i]:
        r2a +=1
        if 'ind_cno_fin_ult1' in predlist[i]:
            r2c += 1
    
    ### For rule 3
    if 'ind_nomina_ult1' in predlist[i] and 'ind_nom_pens_ult1' in predlist[i]:
        r3a +=1
        if 'ind_cno_fin_ult1' in predlist[i]:
            r3c += 1
    
    ### For rule 4
    if 'ind_nomina_ult1' in predlist[i]:
        r4a +=1
        if 'ind_nom_pens_ult1' in predlist[i] and 'ind_cno_fin_ult1' in predlist[i]:
            r4c += 1
            
print("For rule 0: " + str(r0c/r0a) + "\t a: " + str(r0a) + "\t c: " + str(r0c))
print("For rule 1: " + str(r1c/r1a) + "\t a: " + str(r1a) + "\t c: " + str(r1c))
print("For rule 2: " + str(r2c/r2a) + "\t a: " + str(r2a) + "\t c: " + str(r2c))
print("For rule 3: " + str(r3c/r3a) + "\t a: " + str(r3a) + "\t c: " + str(r3c))
print("For rule 4: " + str(r4c/r4a) + "\t a: " + str(r4a) + "\t c: " + str(r4c))


# In[ ]:





# In[ ]:





# In[ ]:




