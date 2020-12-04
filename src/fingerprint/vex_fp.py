import sys
import context
import classes.config
import classes.database
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
np.set_printoptions(threshold=np.nan)

conf = classes.config.Config()
dbt   = classes.database.Database(conf)
common_symbols = set( dbt.distinct_symbol_names() )


for arch, collection in [('x86_64', 'symbols'), ('ARMv7', 'symbols_ARMv7')]:
    #switch to symbols db
    conf.database.collection_name = collection
    db   = classes.database.Database(conf)
    match = { '$match' : { 'arch' : arch } }
    proj = { '$project' : { '_id' : '$name' } }

    query = [ match, proj ]
    res = db.run_mongo_aggregate(query)
    unique_names = set([])
    for i in res:
        unique_names.add( i['_id'] )

    common_symbols = common_symbols.intersection(unique_names)

print(len(common_symbols))


symbols = []
##get symbols
for arch, collection in [('x86_64', 'symbols'), ('ARMv7', 'symbols_ARMv7')]:
    conf.database.collection_name = collection
    db   = classes.database.Database(conf)
    match = { '$match' : { 'arch' : arch, 'name' : { '$in' : list(common_symbols) }  } }
    proj = { '$project' : { 'name' : 1, 'opcode_hash' : 1, 'vex' : { '$concatArrays' : [ '$vex.statements', '$vex.operations', '$vex.expressions' ] }  } }

    query = [ match, proj ]
    res = db.run_mongo_aggregate(query)
    unique_names = set([])
    for i in res:
        s = { 'name' : i['name'], 'opcode_hash' : i['opcode_hash'], 'vex': i['vex'] }
        unique_names.add( s )

print(len(s))
sys.exit()




#print(s[0].vex)
#reduce to unique machine code entries
us = set(map(lambda x: x.opcode_hash, s))


# In[38]:


print(len(us))
ns = []
for oh in us:
    for i in s:
        if i.opcode_hash == oh:
            ns.append(i)
            break
#print(ns)


# In[39]:


selems = list(map(lambda x: { 'name' : x.name, 'arch': x.arch, 'comp': x.compiler, 'opt' : x.optimisation, 'vex': np.hstack([np.array(x.vex['operations']), np.array(x.vex['statements']), np.array(x.vex['expressions'])])}, ns))
df = pd.DataFrame(selems)
#print(df)
#print(df.loc[df['name']=='set_program_name'])


# In[37]:


unique_symbol_names = set(df['name'])
#drop symbols that we don't have multiple of
for i in unique_symbol_names:
    if len(df.loc[df['name'] == i]) == 1:
        df.drop(df.loc[df['name'] == i].index)
print(df)
#create tSNE
#print(np.array(df['vex']))
X = df['vex'].values
#print(len(X))
#print(X[1:3])
#print(np.concatenate(X[:]).reshape(-1, 32))
#print(X[2].reshape(1, -1))


#Y = np.concatenate(X[:]).reshape(-1, 32)
#X_tsne = TSNE(n_components=2).fit_transform(Y)

#print(X_tsne)


# In[ ]:


#with pd.option_context('display.max_rows', 256, 'display.max_columns', 256):
#    print(df.loc[0:1]['vex'][0])

#generate class symbols and color names
#13*7
markers = ['1', '2', '3', '4', 'v', '^', '<', '>', '.', '+', '*', 'h']
colors = ['r', 'g', 'b', 'y', 'c', 'k', 'm']
indicator = {}

#assert(len(markers) * len(colors) >= len(unique_symbol_names))
m, c = 0, 0
for i in unique_symbol_names:
    indicator[i] = [markers[m], colors[c]]
    c += 1
    if c >= len(colors):
        c = 0
        m += 1
        if m >= len(markers):
            break


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_axis_off()
#gsdf = pd.DataFrame(X_tsne, index=gdf.index, columns=['x', 'y'])
#sdf = gsdf.loc[gsdf['name'] == exp_vocab]

#exp_indexes = gdf.loc[gdf['name'].isin(exp_vocab)].index
#sdf = gsdf.loc[exp_indexes]
#ax.scatter(df['x'], df['y'])
ax.grid(False)

for ind, row in df.iterrows():
    if row['name'] not in indicator:
        continue
    mark, col = indicator[row['name']]
    #print(mark, col)
    ax.scatter( X_tsne[ind][0], X_tsne[ind][1], marker=mark, c=col)
    

#for colour, class_name in zip(class_colours, classes):
#    class_indexes = df.loc[ df['class'] == class_name].index
#    ax.scatter( sdf.loc[class_indexes]['x'], sdf.loc[class_indexes]['y'], marker=plt_marker, c=colour, label=class_name, alpha=plt_alpha, s=plt_scale)
                
ax.legend()
#plt.title("t-SNE plot of Symbol Vectors")
plt.legend(loc='lower center', ncol=6, prop={'size': 6})

plt.tight_layout()
#plt.axis('off')
#plt.show()
#fig.savefig()
for fmt in [ 'png']:
    plt.savefig('/tmp/tSNE.VEX.{}'.format(fmt), format=fmt, dpi=900)


# In[40]:


#build KnearestNeighbour Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
neigh = KNeighborsClassifier(n_neighbors=24)




unique_symbol_names = set(df['name'])
class_to_index = { k:v for v, k in enumerate(list(unique_symbol_names)) }
X = df['vex'].values
Y = list(map(lambda x: class_to_index[x], df['name']))

#print(X.shape) == (1340, )
x = [ i for s in X for i in s]
x = np.array(x).reshape(-1, 32)

pca = PCA(n_components=10)
pca_result = pca.fit_transform(x)

#print(x)
#y = np.array(Y)
#x = np.array(X).reshape(-1, 32)
neigh.fit(pca_result, Y)
#print(X)


# In[41]:


df


# In[10]:


#neigh.predict_proba(pca_result[2212])
print(pca_result[2212])
neigh.predict(pca_result[2212].reshape(1, -1))
#neigh.kneighbors(df.iloc[1328]['vex'].reshape(1, -1))


# In[11]:


index_to_class = {k: v for v, k in class_to_index.items()}


# In[12]:


index_to_class[563]


# In[13]:


df.iloc[563]['vex']


# In[14]:


df.iloc[2212]['vex']


# In[42]:


from sklearn.feature_selection import SelectFwe, chi2, SelectKBest
#X_new = SelectFwe(chi2, alpha=0.001).fit_transform(x, Y)
X_new = SelectKBest(chi2, k=16).fit_transform(x, Y)


# In[48]:


X_new.shape


# In[49]:


X_new
import sklearn
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(x, Y)


# In[50]:


from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC


clf = RandomForestClassifier(n_estimators=100)
#clf = GaussianNB()
clf.fit(X_train, Y_train)
pred = clf.predict(X_train)
score = metrics.accuracy_score(Y_train, pred)
print("training accuracy:\t{:0.3f}".format(score) )

pred = clf.predict(X_test)
score = metrics.accuracy_score(Y_test, pred)
print("testing accuracy:\t{:0.3f}".format(score) )


# In[51]:


X_train.shape

