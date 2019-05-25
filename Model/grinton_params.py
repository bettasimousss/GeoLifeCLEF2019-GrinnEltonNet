# -*- coding: utf-8 -*-


class GrintonParams(object):
    
    def  __init__(self,R=24,NC=22,NP=10,NA=1,NBS=10,EMBS=3,WS=5,BN=False,NBPLANTS=20,
    NBMP=[4,3,3,6,3,2,6,4,2,8],NBMA=[46],
    KP=[2,1,1,3,1,1,3,2,1,3],KA=[5],
    NBNP=1055,window=50,EMBNP=50,
    Cnames=['alti','etp','chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6', 'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12', 'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18', 'chbio_19','proxi_eau_fast'],
    Pnames=['awc_top', 'bs_top', 'cec_top','crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top','text'],Anames=['clc'],act="leakyrelu",actfc="relu",actemb=None):
        self.R=R
        self.NC=NC
        self.NP=NP
        self.NA=NA
        self.NBS=NBS
        self.EMBS=EMBS
        self.WS=WS
        self.BN=BN
        self.NBPLANTS=NBPLANTS
        self.NBMP=NBMP
        self.NBMA=NBMA
        self.KP=KP
        self.KA=KA
        self.Cnames=Cnames
        self.Anames=Anames
        self.Pnames=Pnames
        self.window=window
        self.NBNP=NBNP
        self.EMBNP=EMBNP
        
    def update_params(self,alt=1,drop=1,act="leakyrelu",actfc="relu",actemb=None):
        self.topoHydroClim_params={
           "nbchannels":self.NC,
           "imsize":self.R,
           "input_dropout":1,
           "nbalt":alt,
           "conv":{
               "nbfilt":[256,256],
               "fsize":[3,3],
               "cs":[1,1],
               "cp":['same','same']
               },
           "pool":{
               "psize":[2,2],
               "ps":[2,2]
                   },
           "activ":[act,act],  
           "fc":{
               "nbfc":0,
               "nnfc":[],
               "actfc":actfc
                },
            "BN": self.BN,
            "reg":{
               "regtype":[None,None],
               "regparam":[[0.01],[0.01]],
               "dropout":[None,None,None]
               }}
        
        self.pedo_params={
           "nbchannels":self.NP,
           "imsize":self.R,
           "embsize":self.KP,
           "nbmods":self.NBMP,
           "input_dropout":1,
           "nbalt":alt,
           "conv":{
               "nbfilt":[256,256],
               "fsize":[3,3],
               "cs":[1,1],
               "cp":['same','same']
               },
           "pool":{
               "psize":[2,2],
               "ps":[2,2]
                   },
           "activ":[act,act],  ##Non-linear activation after pooling (less computation)
           "fc":{
               "nbfc":0,
               "nnfc":[],
               "actfc":actfc
                },
            "BN": self.BN,
            "reg":{
               "regtype":[None,None],
               "regparam":[[0.01],[0.01]],
               "dropout":[None,None,None],
               "regemb":("l2",[0.01])
               }}
        
        
        self.anthropo_params={
           "nbchannels":self.NA,
           "imsize":self.R,
           "embsize":self.KA,
           "nbmods":self.NBMA, ##0size of array=NA Number of anthropo variables
           "input_dropout":1,
           "nbalt":alt,
           "conv":{
               "nbfilt":[256,256],
               "fsize":[3,3],
               "cs":[1,1],
               "cp":['same','same']
               },
           "pool":{
               "psize":[2,2],
               "ps":[2,2]
                   },
           "activ":[act,act],  ##Non-linear activation after pooling (less computation)
           "fc":{
               "nbfc":0,
               "nnfc":[],
               "actfc":actfc
                },
            "BN": self.BN,
            "reg":{
               "regtype":["l2","l2"],
               "regparam":[[0.01],[0.01]],
               "dropout":[None,None,None],
               "regemb":("l2",[0.01])
               }}
        
        self.ft_params_sep={"nl":2,  ##includes the hidden layers + output
                   "nn":[8192,4096],
                   "nbclass":self.NBPLANTS,
                   "ft_act":actfc,
                    "reg":{
                       "regtype":["l1"],
                       "regparam":[0.05],
                       "dropout":drop
                       }} 
        
        self.ft_params_join={"nl":2,  ##includes the hidden layers + output
                   "nn":[8192,4096],
                   "ft_act":actfc,
                   "nbclass":self.NBPLANTS,
                    "reg":{
                       "regtype":["l1"],
                       "regparam":[0.05],
                       "dropout":drop
                       }} 
        
        self.biogroup_params={"nbnum":self.WS,
                "nl":2,  ##includes only the hidden layers
                "nn":[self.EMBS,self.NBPLANTS],
                "embactiv":actemb,
                "reg":{"regtype":"l2",
                "regparam":[0.1]
                      }
                }
        
        self.bioemb_params={"nbspecies":self.NBNP,
                    "embsize": self.EMBNP,
                    "contextsize":self.window,
                    "embactiv":actemb,
                    "nl":1,  ##includes only the hidden layers
                    "nn":[self.NBPLANTS],
                    "embactiv":actemb,
                    "reg":{"regtype":None,
                    "regparam":[0.1]
                    }
                }
        
        self.ensemble_grinton_params={"meth":False,
                "nl":1,  ##includes only the hidden layers
                "nn":[self.NBPLANTS],
                "activ":actfc,
                "reg":{"regtype":"l2",
                "regparam":[0.1]
                      }
                }
                
        self.joint_grinton_params={"meth":True,
                "nl":1,  ##includes only the hidden layers
                "nn":[self.NBPLANTS],
                "activ":actfc,
                "reg":{"regtype":"l2",
                "regparam":[0.1]
                      }
                }
        self.bio_params=self.bioemb_params #,self.bioemb_params,self.biogroup_params]
        
        self.spat_params_list=[self.topoHydroClim_params,self.anthropo_params,self.pedo_params]
        self.feat_names_list=[self.Cnames,self.Anames,self.Pnames]
        self.trt=[0,1,1]
        self.im_name_list=["climate","landcover","soil"]
