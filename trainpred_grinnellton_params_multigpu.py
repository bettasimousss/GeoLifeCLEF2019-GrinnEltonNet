# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 11:33:22 2019

@author: saras
"""
import sys
import argparse
sys.path.append(".")
import numpy as np
from keras.utils import Sequence
from environmental_raster_glc import PatchExtractor
import pandas as pd
from deep_grinton import GrinnellianNiche, Grinton, EltonianNiche
from grinton_params import GrintonParams
from custom_metrics import topk_accuracy, focal_loss_softmax
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import adam, sgd
#import keras.backend as K
from keras.utils import multi_gpu_model
np.random.seed(4567)


class GlcGenerator(Sequence):

    def __init__(self, extractors, dataset, labels,comps,batch_size,shuffle=True,name="train",folder_np="../Data/retained_np/",window=100,vocab_size=1055,archi=0,runmode=True): ###extractor
        self.extractors = extractors
        self.labels = labels
        self.dataset = dataset
        self.comps= comps
        self.batch_size=batch_size
        self.indices=np.arange(len(self.dataset))
        self.shuffle=shuffle
        #self.extracted=[]
        self.window=window
        self.vocab_size=vocab_size
        self.name=name
        self.archi=archi
        self.runmode=runmode
        if(runmode):
            file_nn="nearest_neighbors.csv"
        else:
            file_nn="nearest_neighbors_test.csv"
        self.dist_np=pd.read_csv(folder_np+file_nn,sep=";",decimal=".",index_col=0)
        
    def __len__(self):
        return int(np.floor(len(self.dataset)/self.batch_size))
    
    def get_cooccur(self,coords):
        #print(coords)
        context=np.concatenate([self.dist_np.query('@lon-1E-6<=Longitude<=@lon+1E-6 & @lat-1E-6<=Latitude<=@lat+1E-6').iloc[0:1,-50:].values for (lon,lat) in coords])
        return context
    
    def __getitem__(self,idx): ###Normalization done at the level of patch extractor 
        #print(self.name + "_" + str(idx))
        indices=self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        if(self.archi==1):  ##grinnell
            tensors=[np.squeeze(np.array([self.extractors[i][self.dataset[j]] for j in indices])) for i in range(len(self.extractors))]
        elif(self.archi==2): ##elton 
            tensors=self.get_cooccur([self.dataset[j] for j in indices])
        else: ##grinton
            cooccur=self.get_cooccur([self.dataset[j] for j in indices])
            tensors=[np.squeeze(np.array([self.extractors[i][self.dataset[j]] for j in indices])) for i in range(len(self.extractors))]
            tensors.append(cooccur)
            
        #self.extracted.append(tensors)
        if self.runmode:
            return tensors, np.expand_dims(self.labels.iloc[indices],axis=1).astype(np.int16)
        else:
            return tensors
    
    def on_epoch_end(self):
        if(self.shuffle):
            np.random.shuffle(self.indices)

class OccurrencePartition(object):
    def __init__(self,mode,dataset):
        self.mode=mode
        self.dataset=dataset
        self.train_idx=None
        self.val_idx=None
        
    def __len__(self):
        return self.dataset.shape[0]
    
    def get_poolsize(self):
        return len(self.dataset["glc19SpId"].unique())
    
    def cross_val_part(self,frac,classstat): 
        if(self.mode=="random"):
            s=np.random.choice(a=2,size=len(self.dataset),replace=True,p=[frac,1-frac])
            self.train_idx=self.dataset.index[np.where(s==0)[0]].tolist()
            self.val_idx=self.dataset.index[np.where(s==1)[0]].tolist()
        elif(self.mode=="stratified"):
            self.train_idx=[]
            self.val_idx=[]
            for c in range(classstat.shape[0]): ##each row is a class
                prev=classstat.loc[c,"prevalences"]
                occurspec=np.where(self.dataset["glc19SpId"]==c)[0]
                np.random.shuffle(occurspec)
                self.train_idx.extend(occurspec[0:max(1,int(prev*frac))])
                self.val_idx.extend(occurspec[-max(1,int(prev*(1-frac))):])
        else:
            print("Wrong partitioning scheme")
            pass
    
    def shuffle_idx(self):
        np.random.shuffle(self.train_idx)
        np.random.shuffle(self.val_idx)

def train(num_epochs=10,batch_size=8,R=16,BN=False,name="TEST",folder_rasters="../Data/rasters_2019",occur_file="../Data/full_occur.csv",taxanames="../Data/test/taxaNameTest.csv",onlytest=0,w=2,opt="sgd",loss="ce",gam=2,LR=0.001,sep=True,decay=True,patience=2,scale=0.2,tolerance=10, metric="acc",Kacc=30,alt=1,drop=1,weighted=True,actemb=None,act="leakyrelu",actfc="relu",
          vocab_size=1055,window=50,EMBNP=100,regional=20,archi=0,grinton_mode=0,gpus=1,init_weights=None,runmode=0):    
    print("Radius ",R,sep=" ")
    print("BatchNorm ",BN)
    clim_vars=['alti','etp','chbio_1', 'chbio_2', 'chbio_3', 'chbio_4', 'chbio_5', 'chbio_6', 'chbio_7', 'chbio_8', 'chbio_9', 'chbio_10', 'chbio_11', 'chbio_12', 'chbio_13', 'chbio_14', 'chbio_15', 'chbio_16', 'chbio_17', 'chbio_18', 'chbio_19','proxi_eau_fast']
    pedo_vars=['awc_top', 'bs_top', 'cec_top','crusting', 'dgh', 'dimp', 'erodi', 'oc_top', 'pd_top','text']
    comps={'clim':(clim_vars,0),'landcover':(["clc"],1),'pedo':(pedo_vars,1)}
    patch_extractors=[]
    if(archi!=2):
        for k in comps.keys():
            trt=comps.get(k)[1]
            varlist=comps.get(k)[0]
            if(trt==0):
                patch_extractor = PatchExtractor(folder_rasters, size=R, verbose=True)
                for v in varlist: 
                    patch_extractor.append(v,normalized=True)
                
                patch_extractors.append(patch_extractor)
            
            else:  ### one patch extractor for each raster (case of categorical variables)
                for v in varlist:
                   pe=PatchExtractor(folder_rasters, size=R, verbose=True)
                   pe.append(v,normalized=False)
                   patch_extractors.append(pe)
    
    allspecies=pd.read_csv(taxanames,sep=";",decimal=".")
    testspecies=allspecies[allspecies.test==True]["glc19SpId"]
    del allspecies
    
    th=1
    dataset=pd.read_csv("../Data/occurrence/full_occur.csv",sep=";",decimal=".")
    if(onlytest==1): ##Train only on test species
        dataset=dataset[dataset["glc19SpId"].isin(testspecies.tolist())]
        th=0
    classes=dataset["glc19SpId"].sort_values().unique()
    prevs=np.array([len(dataset[dataset["glc19SpId"]==c]) for c in classes])
    freq=classes[np.where(prevs>th)[0]]
    dataset=dataset[dataset["glc19SpId"].isin(freq)]
        
    class_codes=np.array(range(0,len(freq)))
    for i in range(len(freq)):
        dataset["glc19SpId"]=dataset["glc19SpId"].replace(freq[i],class_codes[i])    
    
    teststatus=[w*int(x in testspecies.tolist())for x in freq]
    encoding=pd.DataFrame(data=np.stack((freq,class_codes,prevs[np.where(prevs>th)[0]],teststatus),axis=1),columns=["glc19SpId","codes","prevalences","teststatus"])
    encoding.to_csv(name+"_trace_encoding.csv",sep=";",decimal=".",index=False)
    
    partitions=OccurrencePartition("stratified",dataset)
    partitions.cross_val_part(0.9,encoding)
    partitions.shuffle_idx()
    
    ## Example of dataset
    x_train = [tuple(x) for x in (dataset.iloc[partitions.train_idx,:])[["Latitude","Longitude"]].values]
    x_val=[tuple(x) for x in (dataset.iloc[partitions.val_idx,:])[["Latitude","Longitude"]].values]
    
    y_train = dataset.iloc[partitions.train_idx,:]["glc19SpId"]  
    y_val = dataset.iloc[partitions.val_idx,:]["glc19SpId"]
    
    data_train_generator = GlcGenerator(patch_extractors, x_train, y_train, comps, batch_size,shuffle=True,name="train",folder_np="../Data/retained_np/",window=window,vocab_size=vocab_size,archi=archi)
    data_val_generator = GlcGenerator(patch_extractors, x_val, y_val, comps, batch_size,shuffle=False,name="valid",folder_np="../Data/retained_np/",window=window,vocab_size=vocab_size,archi=archi)
    
    gp=GrintonParams(NBPLANTS=partitions.get_poolsize(),R=R,BN=BN,NBNP=vocab_size,window=window,EMBNP=EMBNP)
    gp.update_params(alt=alt,drop=drop,act=act,actfc=actfc,actemb=actemb)
 
    if(archi!=2):
        #### First architecture: Grinnell ####
        Grinnell=GrinnellianNiche(sep,archi)
        if(sep):
            Grinnell.create_grinnell(gp.Anames,gp.Pnames,gp.topoHydroClim_params, gp.pedo_params, gp.anthropo_params, gp.ft_params_sep,gp.spat_params_list,gp.trt,gp.feat_names_list,gp.im_name_list)
        else:
            Grinnell.create_grinnell(gp.Anames,gp.Pnames,gp.topoHydroClim_params, gp.pedo_params, gp.anthropo_params, gp.ft_params_join,gp.spat_params_list,gp.trt,gp.feat_names_list,gp.im_name_list)
        ### Plot model params and architecture ###
        Grinnell.plot_grinnell(name+"_grinnell.png")
        #Grinnell.grinnell.summary()
    if(archi!=1):    
        ### Second architecture: Elton ####
        if(archi==2):
            isfinal=True
        else:
            isfinal=False
        Elton=EltonianNiche(final=isfinal)
        Elton.create_elton(bio_params=gp.bio_params,emb=0)
        #Elton.plot_elton(name+"_elton.png")
        #Elton.elton.summary()
        
    if(archi==1):
        parallel_model=Grinnell.grinnell
    elif(archi==2):
        parallel_model=Elton.elton
    else:
        grinton=Grinton(Grinnell.grinnell,Elton.elton)
        if grinton_mode:
            grinton.create_grinton(gp.ensemble_grinton_params)
        else:
            grinton.create_grinton(gp.joint_grinton_params)
        #grinton.plot_grinton(name+"_grinton.png")
        
        parallel_model=grinton.grinton
    
    parallel_model.summary()

    ### use GPU for data parallelism
    if gpus>1:
        parallel_model=multi_gpu_model(parallel_model, gpus=4)    
    
    if(init_weights!=None):
        parallel_model.load_weights(init_weights)
        
    if(opt=="adam"):
        optimizer=adam(lr=LR)
    else:
        optimizer=sgd(lr=LR, momentum=0.9, nesterov=True)

    if(loss=="fl"):
        obj=focal_loss_softmax(gamma=gam)
    else:
        obj="sparse_categorical_crossentropy"
    
    if(metric=="acc"):
        met="sparse_categorical_accuracy"
    else:
        met=topk_accuracy(Kacc)
        
    parallel_model.compile(optimizer,obj,[met])
    #Grinnell.compile_grinnell(optimizer,obj,[topk_accuracy(3)])
    
    if runmode:
        #### Callbacks ####
        callbcks=[]
        ##TensorBoard
        tbc=TensorBoard(log_dir='./logs_'+name)
        callbcks.append(tbc)
        
        ##Training hyperparameters update => weight decay
        if(decay):
            wdc=ReduceLROnPlateau(monitor='val_loss', factor=scale,patience=patience, min_lr=0.000001, min_delta=0.001)
            esc=EarlyStopping(monitor='val_loss',min_delta=1E-4,patience=tolerance)
            callbcks.append(wdc)
            callbcks.append(esc)
            
        ##Checkpointing the model periodically
        filepath=name+"_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        cpc=ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=5)
        callbcks.append(cpc)
        
        #### Start training #####
        #batch_size=2
        if(onlytest==2):
            classweights=[max(1,encoding.loc[i,"teststatus"]) for i in range(len(encoding))]
        elif(weighted):
            classweights=(1/encoding["prevalences"].values).tolist()
        else:
            classweights=np.ones(len(encoding))
        
        #oi=parallel_model.get_weights()
        #init_weights="elton_standalone_weights.15-6.57.hdf5"
        
        #ol=parallel_model.get_weights()        
        print("Train Mode")
        train_history=parallel_model.fit_generator(generator=data_train_generator,
                                              steps_per_epoch=(len(partitions.train_idx) // batch_size),
                                              epochs=num_epochs,
                                              verbose=1,
                                              validation_data=data_val_generator,
                                              validation_steps=(len(partitions.val_idx) // batch_size),
                                              use_multiprocessing=False,
                                              shuffle=True,
                                              callbacks=callbcks,
                                              class_weight=classweights,
                                              workers=1)

        print("End of training")
        #perform_test=Grinnell.grinnell.evaluate_generator(generator=data_val_generator,verbose=1)
        #trainex=data_train_generator.extracted
        #validex=data_val_generator.extracted
        ### Save model ###
        print("Saving model")
        path=name+".h5"
        parallel_model.save_weights(path)
        
        return train_history
      
    else:
        if onlytest==1:
            encod_file="pretrain/test_encoding.csv"
        else:
            encod_file="pretrain/full_encoding.csv"
            
        predict(parallel_model,patch_extractors,comps,testset="../Data/test/testSet.csv",encoding_file=encod_file,R=R,archi=archi,folder_rasters=folder_rasters,window=window,vocab_size=vocab_size,run_name=name)


def predict(model,patch_extractors,comps,testset="../Data/test/testSet.csv",encoding_file="pretrain/full_encoding.csv",
            R=32,archi=1, folder_rasters="../Data/rasters_2019",
            window=50,vocab_size=1055,run_name="RUNX"):
    
    ### Get training set
    testset=pd.read_csv(testset,sep=";",decimal=".")  
    
    x_test = [tuple(x) for x in (testset)[["Latitude","Longitude"]].values]
    data_test_generator = GlcGenerator(patch_extractors, x_test, [], comps, batch_size=1,shuffle=False,name="train",folder_np="../Data/retained_np/",window=window,vocab_size=vocab_size,archi=archi,runmode=0)
    
    ### Predict
    print("Predict")
    y_predict = model.predict_generator(data_test_generator,steps = len(x_test),verbose=1)
    print("End predict")
    #k=30
    #classes=y_predict.argsort(axis=1)[:,-1*k:][::-1]
    
    ### Reset species encoding 
    encoding=pd.read_csv(encoding_file,sep=";",decimal=".") 
    
#    for c in range(len(encoding)):
#        code=encoding.loc[c,"codes"]
#        glccode=encoding.loc[c,"glc19SpId"]
#        classes[classes==code]=glccode
#
#    classes_df=pd.DataFrame(data=classes,columns=["class_"+str(i) for i in range(k)])
#    predictions=pd.concat([testset,classes_df],axis=1)    
    
    ### Save predictions
    #predictions.to_csv(run_name+"_resultsdf.csv",sep=";",decimal=".")
    format_runs(y_predict,encoding["glc19SpId"].tolist(),testset["glc19TestOccId"].tolist(),50,run_name)
    

def format_runs(y_predict,taxa,occ_ids,n=50,run_name="runx"):
    pred=pd.DataFrame(data=y_predict,columns=taxa)
    pred["glc19TestOccId"]=occ_ids
    
    pred.to_csv(run_name+"_raw_predictions.csv",sep=";",decimal=".")
    
    longt=pred.melt(id_vars=["glc19TestOccId"],var_name="glc19SpId",value_name="Probability")
    
    long_sorted=longt.sort_values(by=["glc19TestOccId","Probability"],ascending=[True,False])
    topn=long_sorted.groupby("glc19TestOccId").head(n)  
    rank=list(range(1,n+1))*pred.shape[0]
    topn["Rank"]=rank
    
    topn[["glc19TestOccId","glc19SpId","Rank","Probability"]].to_csv(run_name+"_glcsubformat.csv",sep=";",decimal=".",index=False)
    
    
def str2bool(v):
    if v.lower() in ('1'):
        return True
    else:
        return False

    
if __name__ == '__main__':
   ### get arguments
    parser=argparse.ArgumentParser()
    parser.add_argument('--sep', dest='sep',default=False,type=str2bool,help='0 for joint and 1 for separate feature extraction')
    parser.add_argument('--lr', dest='LR',default=0.001,type=float,help='Initial learning rate')
    parser.add_argument('--epoch',dest='num_epochs',default=10,type=int,help='Number of epochs')
    parser.add_argument('--bs',dest='batch_size',default=16,type=int,help='Batch size')
    parser.add_argument('--radius',dest='R',default=32,type=int,help='Radius')
    parser.add_argument('--n',dest='name',default="RUN0",type=str,help='Name of run')
    parser.add_argument('--fr',dest='folder_rasters',default="../Data/rasters_2019",type=str,help='Folder where to find rasters')
    parser.add_argument('--fo',dest='occur_file',default="../Data/occurrence/full_occur.csv",type=str,help='CSV file with train occurrence data')
    parser.add_argument('--tn',dest='taxanames',default="../Data/test/taxaNameTest.csv",type=str,help='CSV file with test species names')
    parser.add_argument('--algo',dest='opt',default="adam",type=str,help='Optimizer algorithm: adam or by default sgd')
    parser.add_argument('--obj',dest='loss',default="ce",type=str,help='Loss function: sparse categorical cross-entropy ce or focal loss fl')
    parser.add_argument('--gamma',dest='gam',default=2,type=int,help='Gamma parameter in case loss=fl otherwise this parameter is ignored')   
    parser.add_argument('--decay',dest='decay',default=True,type=str2bool,help='Whether to decay weight in SGD')   
    parser.add_argument('--patience',dest='patience',default=5,type=int,help='Number of epochs to tolerate metric degradation before decaying')   
    parser.add_argument('--tolerance',dest='tolerance',default=10,type=int,help='Number of epochs to decay before stopping early')   
    parser.add_argument('--scale',dest='scale',default=0.2,type=float,help='Scaling of learning rate decay')   
    parser.add_argument('--onlytest',dest='onlytest',default=False,type=str2bool,help='Whether to use only the test species')   
    parser.add_argument('--w',dest='w',default=2,type=int,help='Whether to use only the test species')   
    parser.add_argument('--Kacc',dest='Kacc',default=30,type=int,help='K for TopK accuracy')   
    parser.add_argument('--met',dest='metric',default="acc",type=str,help='Evaluation metric')
    parser.add_argument('--alt',dest='alt',default=1,type=int,help='Number of conv blocks') 
    parser.add_argument('--drop',dest='drop',default=1,type=float,help='Fully connected layer dropout')
    parser.add_argument('--cweight',dest='weighted',default=True,type=str2bool,help='Fully connected layer dropout')
    parser.add_argument('--actemb',dest='actemb',default=None,type=str,help='Activation for embedding layers')
    parser.add_argument('--act',dest='act',default="leakyrelu",type=str,help='Activation for conv layers')
    parser.add_argument('--actfc',dest='actfc',default="relu",type=str,help='Activation for fc layers')
    parser.add_argument('--bnorm',dest='BN',default=False,type=str2bool,help='Batch normalization enabled')
    parser.add_argument('--embnp',dest='EMBNP',default=100,type=int,help='Size of non plant embeddings') 
    parser.add_argument('--regional',dest='regional',default=20,type=int,help='Extension of research for neighbors') 
    parser.add_argument('--wind',dest='window',default=50,type=int,help='Number of neighboring species to consider') 
    parser.add_argument('--archi',dest='archi',default=0,type=int,help='0 for grinton, 1 for grinnell, 2 for elton')
    parser.add_argument('--grtmod',dest='grinton_mode',default=0,type=str2bool,help='0 for joint 1 for ensemble')
    parser.add_argument('--gpus',dest='gpus',default=0,type=int,help='Number of gpus to use')
    parser.add_argument('--init_weights',dest='init_weights',default=None,type=str,help='H5 file with pretrained model weights')
    parser.add_argument('--runmode',dest='runmode',default=1,type=str2bool,help='0 for train 1 for prediction')

    args=parser.parse_args()
    #use_args(**vars(args))
    history=train(**vars(args))
    
