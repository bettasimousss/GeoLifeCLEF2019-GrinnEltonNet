# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:33:54 2019

@author: saras
"""

import h5py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

model_params="Analysis/for_analysis/eltontrans/ELTON_TRANS.hdf5"

f = h5py.File(model_params, 'r')
list(f.keys())

key='model_1'
np_emb=f[key]['embed_nonplant_context']['embeddings:0'][:]

elton_codes="NN/codes_elton.csv"
np_names=pd.read_csv(elton_codes,sep=";",decimal=".",usecols=['taxo', 'code', 'eltoncode'])

np_groups=np_names['code'].str.split('_',1,expand=True).iloc[:,0]
npmetadata=pd.concat([np_names.loc[:,['taxo']],np_groups],axis=1)
npmetadata.to_csv("Analysis/np_metadata.tsv",index=False,header=False,sep="\t")

embeddings=pd.DataFrame(np_emb.T,columns=np_names['taxo'])
embeddings.to_csv("Analysis/np_embeddings.csv",sep=";",decimal=".")

pd.DataFrame(np_emb).to_csv("Analysis/embeddings.tsv",sep="\t",decimal=".",header=False,index=False)
np_names['taxo'].to_csv("Analysis/names.tsv",sep="\t",decimal=".",header=False,index=False)


folder_feat_emb="Analysis/for_analysis/"
clc=pd.read_csv(folder_feat_emb+"clc_emb.tsv",sep="\t",decimal=".",header=None)
clc_assoc=cosine_similarity(clc)
#nc=8
#cahclc=AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
#                    connectivity=None,
#                    linkage='ward', memory=None, n_clusters=nc)
#cahclc.fit(clc)
#clclabels=cahclc.labels_
#
#plt.figure()
#plt.axes([0, 0, 1, 1])
#for l, c in zip(np.arange(cahclc.n_clusters), 'rgbk'):
#    plt.plot(clc[cahclc.labels_ == l].T, c=c, alpha=.5)
#plt.axis('tight')
#plt.axis('off')
#plt.suptitle("AgglomerativeClustering(affinity=%s)" % "cosine", size=20)
#
#    
    
text=pd.read_csv(folder_feat_emb+"text_emb.tsv",sep="\t",decimal=".",header=None)
erodi=pd.read_csv(folder_feat_emb+"erodi_emb.tsv",sep="\t",decimal=".",header=None)
crust=pd.read_csv(folder_feat_emb+"crusting_emb.tsv",sep="\t",decimal=".",header=None)


eltonplantcodes=pd.read_csv("Analysis/for_analysis/ELTON_encoding.csv",sep=";",decimal=".")
glcidnames=pd.read_csv("Data/test/taxaNameTest.csv",sep=";",decimal=".",index_col="glc19SpId")["taxaName"]

names=glcidnames.loc[eltonplantcodes["glc19SpId"].values.tolist()]

o=names.str.split(' ', n=2,expand=True).iloc[:,0:2]
o['fullnames']=o.iloc[:,0]+"_"+o.iloc[:,1]

names.to_csv("Analysis/for_analysis/testplantnames.tsv",sep="\t",header=False,index=False)
o.to_csv("Analysis/for_analysis/testplantnamesfullnom.tsv",sep="\t",header=False,index=False)

plantrespemb=pd.read_csv("Analysis/for_analysis/eltontrans/response_testspecies300.tsv",sep="\t",decimal=".",header=None)
#plantrespemb.to_csv("Analysis/for_analysis/eltontrans/response_testspecies300.tsv",sep="\t",decimal=".",header=False,index=False)

npeffemb=pd.read_csv("Analysis/for_analysis/eltontrans/np_embed300.tsv",sep="\t",decimal=".",header=None)

assoceltontrans=cosine_similarity(plantrespemb,npeffemb)

plt.hist(assoceltontrans)
plt.title("Histogram of associations strength")
plt.show()

assocdf=pd.DataFrame(assoceltontrans,columns=np_names['taxo'].tolist(),index=o['fullnames'].tolist())
rangespec=assocdf.max(axis=1)-assocdf.min(axis=1)

for th in [0.5,0.75,0.8,0.9]:
    assoceltontrans=cosine_similarity(plantrespemb,npeffemb)
    assocdf=pd.DataFrame(assoceltontrans,columns=np_names['taxo'].tolist(),index=o['fullnames'].tolist())
    assocdf[np.abs(assocdf)<th]=0
    
    print("Pos",len(np.where(assocdf<0)[0]))
    print("Neg",len(np.where(assocdf>0)[0]))
    
    signif_assoc=np.stack(np.where(assocdf!=0),axis=1)
    rows=np.array([assocdf.index.tolist()[s] for s in signif_assoc[:,0].tolist()])
    cols=np.array([assocdf.columns.tolist()[s] for s in signif_assoc[:,1].tolist()])
    
    pairs=np.stack([rows,cols],axis=1)
    strengths=np.array([assocdf.loc[x,y] for (x,y) in pairs]).reshape(len(pairs),1)
    
    edge_list=np.concatenate([pairs,strengths],axis=1)
    
    edgelist=pd.DataFrame(edge_list,columns=['source','target','weight'])
    edgelist.to_csv("Analysis/association_np_p.csv",sep=";",decimal=".",index=False)
    
    ####### Visualize graph ########
    edgelist['weight']=edgelist['weight'].astype(float)
    G = nx.from_pandas_edgelist(edgelist, edge_attr=True)
    npset=set(edgelist['source'].tolist())
    print(len(npset))
    pset=set(edgelist['target'].tolist())
    print(len(pset))
    print(len(G.nodes)==(len(npset)+len(pset)))
    
    print(len(G.edges))
    weights=edgelist['weight'].tolist()
    pos = nx.bipartite_layout(G, nodes=npset)
    
#    plt.figure(figsize=(24,24)) 
#    nx.draw(G,pos,node_size=100,font_size=16,with_labels=True,edge_color=weights,edge_cmap=plt.cm.coolwarm)
#    
#    plt.savefig("Analysis/association_"+str(th)+".png")
