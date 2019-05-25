# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:51:07 2019

@author: saras
"""

import pandas as pd
import numpy as np

npl="Data/temp/noPlant.csv"

cols=['kingdom', 'phylum', 'class','order', 'family', 'genus', 'species', 
 'Latitude','Longitude', 'coordinateUncertaintyInMeters','day', 'month', 
 'year','scName', 'glc19SpId']

nonplants=pd.read_csv(npl,sep=";",decimal=".",usecols=cols)
nonplants[nonplants['month']!=nonplants['month']]=-1
nonplants[nonplants['month']=='month']=-1
nonplants['month']=nonplants['month'].astype(int)

global_codes=[]

### Process fungi ###
th=0.5
fungi=nonplants[nonplants['kingdom']=="Fungi"]
sel_wbk=pd.read_csv("Data/expert_fungi_selection.csv",sep=";",decimal=".").dropna()
kept=sel_wbk.query('Keep >= @th')
kept['code']=["F_"+str(i) for i in range(len(kept))]
kept.columns=['genus','Keep','code']
genera=kept['genus']
#code=kept['code']
fungi=fungi[fungi["genus"].isin(genera.tolist())]
fungi_coded=pd.merge(fungi,kept,on='genus')

codes_fungi_df=kept[['genus','code']]
fungi_group=fungi_coded.drop_duplicates(subset=['Longitude','Latitude','code'])
sel_cols=['genus','Latitude', 'Longitude','glc19SpId', 'Keep', 'code','eltoncode']
#fungi_group.loc[:,sel_cols].to_csv("Data/retained_np/fungi_occurrence.csv",sep=";",decimal=".",index=True)

### Process arthropods ###
arthro=nonplants[nonplants['phylum']=="Arthropoda"]
sel_arthro=pd.read_csv("Data/expert_arthropods_selection.csv",sep=";",decimal=".").dropna()
kept_order_arthro=sel_arthro.query('Keep!=0')
#kept_order_arthro['code']=["AO_"+str(i) for i in range(len(kept_order_arthro))]
#kept_family_arthro=sel_arthro.query('Keep==2')
#kept_family_arthro['code']=["AF_"+str(i) for i in range(len(kept_family_arthro))]
arthro=arthro[arthro["order"].isin(kept_order_arthro["order"].tolist())]

family_arthro=arthro['family'].dropna().sort_values().unique().tolist()
codes_arthro=["A_"+str(i) for i in range(len(family_arthro))]

code_arthro_df=pd.DataFrame(data=np.array([family_arthro,codes_arthro]).T,columns=["family","code"])

arthro_coded=pd.merge(arthro,code_arthro_df,on='family')
arthro_group=arthro_coded.drop_duplicates(subset=['Longitude','Latitude','code'])

#arthro_group.loc[:,['family','Latitude', 'Longitude','glc19SpId', 'code','eltoncode']].to_csv("Data/retained_np/arthro_occurrence.csv",sep=";",decimal=".",index=True)

### Process Chordata/Aves ###
toremove=['Acridotheres',
'Aegypius',
'Agapornis',
'Agapornis',
'Aix',
'Aix',
'Alcedo',
'Alle',
'Alopochen']
aves=nonplants[nonplants['class']=="Aves"].query('3<=month<=7')
aves_group=aves.drop_duplicates(['Longitude','Latitude','genus'])
aves_group=aves_group[~aves_group['genus'].isin(toremove)]

unique_genuses=aves_group['genus'].dropna().sort_values().unique().tolist()
codes_aves=["B_"+str(i) for i in range(len(unique_genuses))]
codes_aves_df=pd.DataFrame(data=np.array([unique_genuses,codes_aves]).T,columns=["genus","code"])
aves_group_coded=pd.merge(aves_group,codes_aves_df,on='genus')

#aves_group_coded.loc[:,['genus','Latitude','Longitude','glc19SpId','code','eltoncode']].to_csv("Data/retained_np/aves_occurrence.csv",sep=";",decimal=".",index=True)

### Process Chordata/Amphibia ###
amph=nonplants[nonplants['class']=="Amphibia"]
amph_group=amph.drop_duplicates(['Longitude','Latitude','genus'])
unique_g_amph=amph['genus'].dropna().sort_values().unique().tolist()
codes_amph=["W_"+str(i) for i in range(len(unique_g_amph))]
codes_amph_df=pd.DataFrame(data=np.array([unique_g_amph,codes_amph]).T,columns=["genus","code"])
amph_group_coded=pd.merge(amph_group,codes_amph_df,on='genus')

#amph_group_coded.loc[:,['genus','Latitude','Longitude','glc19SpId','code','eltoncode']].to_csv("Data/retained_np/amph_occurrence.csv",sep=";",decimal=".",index=True)

### Process Chordata/Reptilia ###
rept=nonplants[nonplants['class']=="Reptilia"]
rept_group=rept.drop_duplicates(['Longitude','Latitude','genus'])
unique_g_rept=rept_group['genus'].dropna().sort_values().unique().tolist()
codes_rept=["R_"+str(i) for i in range(len(unique_g_rept))]
codes_rept_df=pd.DataFrame(data=np.array([unique_g_rept,codes_rept]).T,columns=["genus","code"])
rept_group_coded=pd.merge(rept_group,codes_rept_df,on='genus')

#rept_group_coded.loc[:,['genus','Latitude','Longitude','glc19SpId','code','eltoncode']].to_csv("Data/retained_np/rept_occurrence.csv",sep=";",decimal=".",index=True)

### Process Chordata/Mammalia ###
mamma=nonplants[nonplants['class']=="Mammalia"]
mamma_group=mamma.drop_duplicates(['Longitude','Latitude','genus'])
unique_g_mamma=mamma_group['genus'].dropna().sort_values().unique().tolist()
codes_mamma=["M_"+str(i) for i in range(len(unique_g_mamma))]
codes_mamma_df=pd.DataFrame(data=np.array([unique_g_mamma,codes_mamma]).T,columns=["genus","code"])
mamma_group_coded=pd.merge(mamma_group,codes_mamma_df,on='genus')

#mamma_group_coded.loc[:,['genus','Latitude','Longitude','glc19SpId','code','eltoncode']].to_csv("Data/retained_np/mammal_occurrence.csv",sep=";",decimal=".",index=True)
for x in [codes_amph_df,code_arthro_df,codes_aves_df,codes_fungi_df,codes_mamma_df,codes_rept_df]:
    x.columns=['taxo','code']
code_taxo_df=pd.concat([codes_amph_df,code_arthro_df,codes_aves_df,codes_fungi_df,codes_mamma_df,codes_rept_df],axis=0,ignore_index=True)
code_taxo_df['eltoncode']=np.arange(0,len(code_taxo_df))

code_taxo_df.to_csv("Data/retained_np/codes_elton.csv",sep=";",decimal=".",index=True)

code_taxo_df.index=code_taxo_df.code
#### Apply absolute elton codes to all files and resaving ###
fungi_group['eltoncode']=[code_taxo_df.loc[c,'eltoncode'] for c in fungi_group['code']]
arthro_group['eltoncode']=[code_taxo_df.loc[c,'eltoncode'] for c in arthro_group['code']]
aves_group_coded['eltoncode']=[code_taxo_df.loc[c,'eltoncode'] for c in aves_group_coded['code']]
amph_group_coded['eltoncode']=[code_taxo_df.loc[c,'eltoncode'] for c in amph_group_coded['code']]
mamma_group_coded['eltoncode']=[code_taxo_df.loc[c,'eltoncode'] for c in mamma_group_coded['code']]
rept_group_coded['eltoncode']=[code_taxo_df.loc[c,'eltoncode'] for c in rept_group_coded['code']]




#### Getting the different phyla into different files ###
#aves=nonplants[nonplants['class']=="Aves"].query('3<=month<=7')
#amph=nonplants[nonplants['class']=="Amphibia"]
#rept=nonplants[nonplants['class']=="Reptilia"]
#mamm=nonplants[nonplants['class']=="Mammalia"]
#
#### Saving to separate files for selection ###
#aves.to_csv("Aves.csv",sep=";",decimal=".")
#amph.to_csv("Amphibia.csv",sep=";",decimal=".")
#rept.to_csv("Reptilia.csv",sep=";",decimal=".")
#mamm.to_csv("Mammalia.csv",sep=";",decimal=".")

#obsaves=aves['month'].unique().astype(str)
