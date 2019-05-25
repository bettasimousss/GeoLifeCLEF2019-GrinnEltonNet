# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 19:19:01 2019

@author: saras
"""

import pandas as pd
import geopy.distance as gd
from joblib import Parallel, delayed
import multiprocessing
import numpy as np

deglon=111.03
deglat=85.39
maxdist=250

def border_ids(up,maxdist):
    lon=up[1]
    lat=up[0]
    
    minlon=lon-maxdist/deglon
    minlat=lat-maxdist/deglat
    maxlon=lon+maxdist/deglon
    maxlat=lat+maxdist/deglat
    
    return minlon, maxlon, minlat, maxlat
 
class NeighborExtractor(object):
    def __init__(self, folder_np="../Data/retained_np/",dist_np=[5,5,5,5,5,5],window=50,regional=[5,5,5,5,5,5]):
        self.folder_np = folder_np
        self.nonp=dict(
                amph=pd.read_csv(folder_np+"amph_occurrence.csv",sep=";",decimal=".",usecols=['Latitude','Longitude','eltoncode'],dtype={'Longitude': float,'Latitude':float,'eltoncode':int}),
                arthro=pd.read_csv(folder_np+"arthro_occurrence.csv",sep=";",decimal=".",usecols=['Latitude','Longitude','eltoncode'],dtype={'Longitude': float,'Latitude':float,'eltoncode':int}),
                aves=pd.read_csv(folder_np+"aves_occurrence.csv",sep=";",decimal=".",usecols=['Latitude','Longitude','eltoncode'],dtype={'Longitude': float,'Latitude':float,'eltoncode':int}),
                fungi=pd.read_csv(folder_np+"fungi_occurrence.csv",sep=";",decimal=".",usecols=['Latitude','Longitude','eltoncode'],dtype={'Longitude': float,'Latitude':float,'eltoncode':int}),
                mammal=pd.read_csv(folder_np+"mammal_occurrence.csv",sep=";",decimal=".",usecols=['Latitude','Longitude','eltoncode'],dtype={'Longitude': float,'Latitude':float,'eltoncode':int}),
                rept=pd.read_csv(folder_np+"rept_occurrence.csv",sep=";",decimal=".",usecols=['Latitude','Longitude','eltoncode'],dtype={'Longitude': float,'Latitude':float,'eltoncode':int}))
        self.dist_np=np.array(dist_np)
        self.taxo=["amph","arthro","aves","fungi","mammal","rept"]
        if(type(regional)==list):
            self.regional=regional
        else:
            self.regional=np.ones(len(dist_np))*regional
        self.window=window
        
    def get_cooccur(self,coords):
        #print(coords)  
        neighbors=[]  
        dist_np=self.dist_np.copy()
        while(len(neighbors)==0):
            for t in range(len(dist_np)):
                min_lon,max_lon,min_lat,max_lat=border_ids(coords,dist_np[t])
                neighbors.extend(self.nonp.get(self.taxo[t]).query('@min_lon<Longitude<@max_lon & @min_lat<Latitude<@max_lat')['eltoncode'].values.tolist())
            dist_np=np.multiply(dist_np,self.regional)
            
        context=np.random.choice(neighbors,size=self.window)
        return context, dist_np
         
if __name__ == '__main__':
    ne=NeighborExtractor("Data/retained_np/",window=50,regional=2,dist_np=[1.,1.,1.,1.,1.,1.])
    data=pd.read_csv("Data/test/testSet.csv",sep=";",decimal=".",usecols=["Latitude","Longitude"])[["Latitude","Longitude"]]
    dataset=[tuple(x) for x in data.drop_duplicates().values]
    nns=[]
    radiuses=[]
    for p in dataset:
        neighbors, rad=ne.get_cooccur(p)
        nns.append(neighbors)
        radiuses.append(rad)
        
    contexts=pd.DataFrame(np.stack(nns))
    coords=pd.DataFrame(np.array(dataset),columns=['Longitude','Latitude'])
    nneighbors=pd.concat([coords,contexts],axis=1)
    nneighbors.to_csv("nearest_neighbors_test2.csv",sep=";",decimal=".",index=True)
    
    precis=pd.DataFrame(np.stack(radiuses),columns=ne.taxo)
    precis.to_csv("maxradiusused_test2.csv",sep=";",decimal=".",index=True)
    
#    ncores=multiprocessing.cpu_count()
#    l=len(ne.u_point)
#    dpc=int(l/ncores)
#    pool=multiprocessing.Pool()
#    um=pool.map(ne.within_radius_idx,[ne.u_point.iloc[i*dpc:(i+1)*dpc,:] for i in range(ncores)])
#    
#    merged_df=pd.DataFrame.from_dict(um)
#    merged_df.to_csv("sel_nn_nonplants0.csv",sep=";",decimal=".")
    
