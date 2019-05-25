import sys
import pandas as pd
sys.path.append("Support_Code/")

#from environmental_raster_glc import Raster, PatchExtractor


def load_occurrences(keep_untrusted=False):
    gbif_plants=pd.read_csv("Data/occurrence/GLC_2018.csv",sep=";",decimal=",",
                            usecols=['Longitude','Latitude','glc19SpId','scName'],
                            dtype = {'Longitude': str,'Latitude': str,'glc19SpId':str,'scName':str})
    PN_complete=pd.read_csv("Data/occurrence/PL1718_complete.csv",sep=";",decimal=",",
                            usecols=['Longitude','Latitude','glc19SpId','scName'],
                            dtype = {'Longitude': str,'Latitude': str,'glc19SpId':str,'scName':str})
    PN_trusted=pd.read_csv("Data/occurrence/PL1718_trusted.csv",sep=";",decimal=",",
                           usecols=['Longitude','Latitude','glc19SpId','scName'],
                           dtype = {'Longitude': str,'Latitude': str,'glc19SpId':str,'scName':str})
    
    occur=[gbif_plants,PN_trusted]
    if(keep_untrusted):
        occur.append(PN_complete)
    return pd.concat(occur,sort=False)


occur_trusted=load_occurrences()
occur_full=load_occurrences(True)

occur_trusted.Longitude=occur_trusted.Longitude.astype(float)
occur_trusted.Latitude=occur_trusted.Latitude.astype(float)
occur_full.Longitude=occur_full.Longitude.astype(float)
occur_full.Latitude=occur_full.Latitude.astype(float)

occur_trusted.to_csv("Data/occur_trusted.csv",sep=";",decimal=".",index=False)
occur_full.to_csv("Data/occur_full.csv",sep=";",decimal=".",index=False)

#
gbif_nonplants=pd.read_csv("Data/occurrence/noPlant.csv",sep=";",decimal=",",
                               usecols=['Longitude','Latitude','glc19SpId','scName',
                                        'kingdom', 'phylum', 'class','order', 'family', 'genus', 'species'],
                               dtype = str)