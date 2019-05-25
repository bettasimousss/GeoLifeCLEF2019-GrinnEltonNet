# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:33:30 2019

@author: saras
"""

####################################################################################################################
                                    ###GRINNELL###
####################################################################################################################
import tensorflow as tf
import numpy as np
tf.set_random_seed(1234)
np.random.seed(100)
import keras.backend as K

# Multiple Outputs
from keras.models import Model
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import glorot_normal
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Embedding, Concatenate, Activation, Dropout, Lambda, Average, BatchNormalization
from keras.layers import LeakyReLU, ELU

def get_regularizer(regtype,regparams):
    if(regtype=="l1"):
        kr=l1(l=regparams[0])
    elif(regtype=="l2"):
        kr=l2(l=regparams[0])
    elif(regtype=="l1_l2"):
        kr=l1_l2(l1=regparams[0],l2=regparams[1])
    else:
        kr=None 
    return kr

def cnn_comp(spat_params,feat_name,extract=True):  ##CNN alternating convolution and pooling layers
    imdimensions=spat_params["imsize"]
    imchannels=spat_params["nbchannels"]
    BN=spat_params["BN"]
    
    nb_alt=spat_params["nbalt"]  ##Number of convolution, pooling alternations

    conv_params=spat_params["conv"]
    pool_params=spat_params["pool"]
    
    fc_params=spat_params["fc"]
    
    reg=spat_params["reg"]
    
    krcnn=get_regularizer(reg.get("regtype")[0],reg.get("regparam")[0])
    krfc=get_regularizer(reg.get("regtype")[1],reg.get("regparam")[1])

    ### 1.Input
    input_raster=Input(shape=(imdimensions,imdimensions,imchannels),name="in_"+feat_name,dtype=tf.float32)
    
    ### 2.Convolutions + Pooling
    if(reg.get("dropout")[0]!=None):
        prevalt=Dropout(rate=reg.get("dropout")[0])(input_raster)
    else:
        prevalt=input_raster
    
    if(extract):
        for i in range(nb_alt):
            ###CONV###
            prevalt=Conv2D(conv_params.get("nbfilt")[i], kernel_size=conv_params.get("fsize")[i], strides=conv_params.get("cs")[i], padding=conv_params.get("cp")[i],kernel_regularizer=krcnn,name=feat_name+"_conv_0"+str(i+1))(prevalt)
            prevalt=Conv2D(conv_params.get("nbfilt")[i], kernel_size=conv_params.get("fsize")[i], strides=conv_params.get("cs")[i], padding=conv_params.get("cp")[i],kernel_regularizer=krcnn,name=feat_name+"_conv_1"+str(i+1))(prevalt)
            ###BatchNorm###
            if BN: prevalt=BatchNormalization()(prevalt)
            ###Pooling###
            prevalt=MaxPool2D(pool_size=pool_params.get("psize")[i], strides=pool_params.get("ps")[i], name=feat_name+"_pool_"+str(i+1))(prevalt)
            ###Nonlinearity###
            if(spat_params["activ"][i]=="leakyrelu"):
                prevalt=LeakyReLU()(prevalt)
            elif(spat_params["activ"][i]=="elu"):
                            prevalt=ELU()(prevalt)
            else:  ##relu or linear
                prevalt=Activation(spat_params["activ"][i])(prevalt)
            
            if(reg.get("dropout")[1]!=None):
                prevalt=Dropout(rate=reg.get("dropout")[1])(prevalt)            
    
        ### 3. Flattening layer
        prevalt=Flatten()(prevalt)
        
        ### 4. Fully connected layer
        for i in range(fc_params.get("nbfc")):
            if(spat_params["activ"][i]=="leakyrelu"):
                prevalt=Dense(fc_params.get("nnfc")[i],kernel_regularizer=krfc,name=feat_name+"_fc_"+str(i+1))(prevalt)
                prevalt=LeakyReLU()(prevalt)
            elif(spat_params["activ"][i]=="elu"):
                prevalt=Dense(fc_params.get("nnfc")[i],kernel_regularizer=krfc,name=feat_name+"_fc_"+str(i+1))(prevalt)
                prevalt=ELU()(prevalt)
            else:  ##relu or linear
                prevalt=Dense(fc_params.get("nnfc")[i], activation=fc_params.get("actfc"),kernel_regularizer=krfc,name=feat_name+"_fc_"+str(i+1))(prevalt)
            if(reg.get("dropout")[2]!=None):
                prevalt=Dropout(rate=reg.get("dropout")[2])(prevalt)
    
    cnn_model=Model(input_raster,prevalt)
    
    return cnn_model

def embcnn_comp(spat_params,feat_names,im_name,extract=True):  ##Embedding first then CNN alternating convolution and pooling layers
    imdimensions=spat_params["imsize"]
    imchannels=spat_params["nbchannels"]
    BN=spat_params["BN"]
    
    nb_alt=spat_params["nbalt"]  ##Number of convolution, pooling alternations
    conv_params=spat_params["conv"]
    pool_params=spat_params["pool"]
    activs=spat_params["activ"]
    fc_params=spat_params["fc"]
    reg=spat_params["reg"]
    
    krcnn=get_regularizer(reg.get("regtype")[0],reg.get("regparam")[0])
    krfc=get_regularizer(reg.get("regtype")[1],reg.get("regparam")[1])
    kremb=get_regularizer(reg.get("regemb")[0],reg.get("regemb")[1])
    
    nbmods=spat_params["nbmods"]
    embsizes=spat_params["embsize"]

    ### 1.Input
    inputs=[]
    embeddings=[]
    for v in range(imchannels):
        in_cat=Input(shape=(imdimensions,imdimensions),name=feat_names[v],dtype=tf.int32)
        embed_cat=Embedding(input_dim=nbmods[v],input_length=(imdimensions,imdimensions),
                            output_dim=embsizes[v],name="embed_"+feat_names[v],
                            embeddings_initializer=glorot_normal(),
                            embeddings_regularizer=kremb)(in_cat)
        
        embeddings.append(embed_cat)
        inputs.append(in_cat)
    
    ### 2.Convolutions + Pooling
    if(imchannels>1):
        raster_embeddings=Concatenate()(embeddings)
    else:
        raster_embeddings=embeddings[0]
        
    ### Embeddings dropout
    if(reg.get("dropout")[0]!=None):
        prevalt=Dropout(rate=reg.get("dropout")[0])(raster_embeddings)
    else:
        prevalt=raster_embeddings
    
    if(extract):
        for i in range(nb_alt):
            prevalt=Conv2D(conv_params.get("nbfilt")[i], kernel_size=conv_params.get("fsize")[i], strides=conv_params.get("cs")[i], padding=conv_params.get("cp")[i],kernel_regularizer=krcnn,name=im_name+"_conv_0"+str(i+1))(prevalt)
            prevalt=Conv2D(conv_params.get("nbfilt")[i], kernel_size=conv_params.get("fsize")[i], strides=conv_params.get("cs")[i], padding=conv_params.get("cp")[i],kernel_regularizer=krcnn,name=im_name+"_conv_1"+str(i+1))(prevalt)
            if BN: prevalt=BatchNormalization()(prevalt)
            prevalt=MaxPool2D(pool_size=pool_params.get("psize")[i], strides=pool_params.get("ps")[i], name=im_name+"_pool_"+str(i+1))(prevalt)
            
            if(activs[i]=="leakyrelu"):
                prevalt=LeakyReLU()(prevalt)
            elif(activs[i]=="elu"):
                prevalt=ELU()(prevalt)
            else:
                prevalt=Activation(activs[i])(prevalt)

            if(reg.get("dropout")[1]!=None):
                prevalt=Dropout(rate=reg.get("dropout")[1])(prevalt)        
            
        ### 3. Flattening layer
        prevalt=Flatten()(prevalt)
        
        ### 4. Fully connected layer
        for i in range(fc_params.get("nbfc")):
            if(fc_params.get("actfc")=="leakyrelu"):
                prevalt=Dense(fc_params.get("nnfc")[i],kernel_regularizer=krfc,name=im_name+"_fc_"+str(i+1))(prevalt)
                prevalt=LeakyReLU()(prevalt)
            elif(fc_params.get("actfc")=="elu"):
                prevalt=Dense(fc_params.get("nnfc")[i],kernel_regularizer=krfc,name=im_name+"_fc_"+str(i+1))(prevalt)
                prevalt=ELU()(prevalt)
            else:
                prevalt=Dense(fc_params.get("nnfc")[i], activation=fc_params.get("actfc"),kernel_regularizer=krfc,name=im_name+"_fc_"+str(i+1))(prevalt)
            if(reg.get("dropout")[2]!=None):
                prevalt=Dropout(rate=reg.get("dropout")[2])(prevalt)
        
    cnn_model=Model(inputs,prevalt)
    
    return cnn_model

def emb_merg_cnn(spat_params_list,trt,feat_names_list,im_name_list,feat_name="ClimSoilLand"):
    ###input components###
    inputs=[]
    out_inc=[]
    for i in range(len(trt)):
        if(trt[i]==1):
            inc=embcnn_comp(spat_params_list[i],feat_names_list[i],im_name_list[i],False)
        else:
            inc=cnn_comp(spat_params_list[i],im_name_list[i],False)
            
        if(type(inc.input)==list):
            inputs.extend(inc.input)
        else:
            inputs.append(inc.input)
        
        out_inc.append(inc.output)
    
    ### Concat to form a single thick patch ###
    prevalt=Concatenate()(out_inc)
    
    ### Extract features using first params ###
    spat_params=spat_params_list[0]
    nb_alt=spat_params["nbalt"]  ##Number of convolution, pooling alternations
    BN=spat_params["BN"]
    conv_params=spat_params["conv"]
    pool_params=spat_params["pool"]
    
    fc_params=spat_params["fc"]
    
    reg=spat_params["reg"]
    
    krcnn=get_regularizer(reg.get("regtype")[0],reg.get("regparam")[0])
    krfc=get_regularizer(reg.get("regtype")[1],reg.get("regparam")[1])
    
    for i in range(nb_alt):
        ###CONV###
        prevalt=Conv2D(conv_params.get("nbfilt")[i], kernel_size=conv_params.get("fsize")[i], strides=conv_params.get("cs")[i], padding=conv_params.get("cp")[i],kernel_regularizer=krcnn,name=feat_name+"_conv_0"+str(i+1))(prevalt)
        prevalt=Conv2D(conv_params.get("nbfilt")[i], kernel_size=conv_params.get("fsize")[i], strides=conv_params.get("cs")[i], padding=conv_params.get("cp")[i],kernel_regularizer=krcnn,name=feat_name+"_conv_1"+str(i+1))(prevalt)
        ###BatchNorm###
        if BN: prevalt=BatchNormalization()(prevalt)
        ###Nonlinearity###
        ###Pooling###
        prevalt=MaxPool2D(pool_size=pool_params.get("psize")[i], strides=pool_params.get("ps")[i], name=feat_name+"_pool_"+str(i+1))(prevalt)

        if(spat_params["activ"][i]=="leakyrelu"):
            prevalt=LeakyReLU()(prevalt)
        elif(spat_params["activ"][i]=="elu"):
            prevalt=ELU()(prevalt)
        else:
            prevalt=Activation(spat_params["activ"][i])(prevalt)
        
        if(reg.get("dropout")[1]!=None):
            prevalt=Dropout(rate=reg.get("dropout")[1])(prevalt)
        
    
    ### 3. Flattening layer
    prevalt=Flatten()(prevalt)
        
    ### 4. Fully connected layer
    for i in range(fc_params.get("nbfc")):
        if(fc_params.get("actfc")=="leakyrelu"):
            prevalt=Dense(fc_params.get("nnfc")[i],kernel_regularizer=krfc,name=feat_name+"_fc_"+str(i+1))(prevalt)
            prevalt=LeakyReLU()(prevalt)
        elif(fc_params.get("actfc")=="elu"):
            prevalt=Dense(fc_params.get("nnfc")[i],kernel_regularizer=krfc,name=feat_name+"_fc_"+str(i+1))(prevalt)
            prevalt=ELU()(prevalt)
        else:
            prevalt=Dense(fc_params.get("nnfc")[i], activation=fc_params.get("actfc"),kernel_regularizer=krfc,name=feat_name+"_fc_"+str(i+1))(prevalt)
        if(reg.get("dropout")[2]!=None):
            prevalt=Dropout(rate=reg.get("dropout")[2])(prevalt)
     
    return Model(inputs,prevalt)
    
def featTransf_merge(ft_params,fe_comps,out_activ=True): ##Transforming extracted features (shared component)
    fe_ins=[]
    fe_out=[]
    for c in fe_comps:
        if(type(c.input)==list):
            fe_ins.extend(c.input)
        else:
            fe_ins.append(c.input)
        fe_out.append(c.output)
    
    if(len(fe_out)>1):
        prev=Concatenate()(fe_out)
    else:
        prev=fe_out[0]
    reg=get_regularizer(ft_params.get("reg").get("regtype"),ft_params.get("reg").get("regparam"))
    for i in range(ft_params.get("nl")):
        if(ft_params.get("ft_act")=="leakyrelu"):
            prev=Dense(ft_params.get("nn")[i],name="shared_fc_"+str(i),kernel_regularizer=reg)(prev)
            prev=LeakyReLU()(prev)
        elif(ft_params.get("ft_act")=="elu"):
            prev=Dense(ft_params.get("nn")[i],name="shared_fc_"+str(i),kernel_regularizer=reg)(prev)
            prev=ELU()(prev)
        else:
            prev=Dense(ft_params.get("nn")[i], activation=ft_params.get("ft_act"),name="shared_fc_"+str(i),kernel_regularizer=reg)(prev)
        if(i<(ft_params.get("nl")-1)):
            if(ft_params.get("reg").get("dropout")!=None):
                prev=Dropout(ft_params.get("reg").get("dropout"))(prev)
    
    if(out_activ):
        out=Dense(ft_params.get("nbclass"),activation="softmax",name="shared_fc_out")(prev)
    else:
        out=prev
    m=Model(fe_ins,out)
    return(m)
    

####################################################################################################################
                                            ### ELTON ###
####################################################################################################################
def group_interaction_comp(bio_params):  ###As a FC_NN
    #nbs=bio_params["nbspecies"]
    act=bio_params["embactiv"]
    ws=bio_params["nbnum"]  ##each element corresponds to the neighboring counts of a group of species
    reg=get_regularizer(bio_params.get("reg").get("regtype"),bio_params.get("reg").get("regparam"))
    
    in_bio=Input(shape=(ws,),name="NonPlants",dtype=tf.float32)

    prev=in_bio
    for i in range(bio_params.get("nl")):
        prev=Dense(bio_params.get("nn")[i], activation=act,name="interaction_"+str(i),kernel_regularizer=reg)(prev)
    
    inmodel=Model(in_bio,prev)
    
    return inmodel

### Context embeddings for each non plant species ###
def embed_interaction_comp(bio_params,final=False):
    nbs=bio_params["nbspecies"]
    ws=bio_params["contextsize"] 
    es=bio_params["embsize"]
    
    act=bio_params["embactiv"]
    reg=get_regularizer(bio_params.get("reg").get("regtype"),bio_params.get("reg").get("regparam"))
    
    in_context=Input(shape=(ws,),name="NonPlants",dtype=tf.int32)
    emb_context=Embedding(input_dim=nbs,
                            output_dim=es,name="embed_nonplant_context",
                            embeddings_initializer=glorot_normal(),
                            embeddings_regularizer=reg)(in_context)
    
    prev=Lambda(lambda xin: K.mean(xin, axis=1))(emb_context)
    
    ###Non linear transformation of the embeddings
    for i in range(bio_params.get("nl")):
        prev=Dense(bio_params.get("nn")[i], activation=act,name="interaction_"+str(i),kernel_regularizer=reg)(prev)

    if final:
        prev=Activation("softmax")(prev)
    model=Model(in_context,prev)
    
    return model

def tfidf_interaction_comp(bio_params):
    nbs=bio_params["nbspecies"]
    ws=bio_params["contextsize"] 
    es=bio_params["embsize"]
    
    act=bio_params["embactiv"]
    reg=get_regularizer(bio_params.get("reg").get("regtype"),bio_params.get("reg").get("regparam"))
    
    in_context=Input(shape=(ws,),name="NonPlants",dtype=tf.int32)
    in_tf=Input(shape=(ws,),name="TFIDF",dtype=tf.float32)
    
    emb_context=Embedding(input_dim=nbs,
                            output_dim=es,name="embed_nonplant_context",
                            embeddings_initializer=glorot_normal(),
                            embeddings_regularizer=reg)(in_context)
    
    w_emb_context=Lambda(lambda v: K.expand_dims(v[0])*v[1])([in_tf,emb_context])
    prev=Lambda(lambda xin: K.mean(xin, axis=1))(w_emb_context)
    
    ###Non linear transformation of the embeddings
    for i in range(bio_params.get("nl")):
        prev=Dense(bio_params.get("nn")[i], activation=act,name="interaction_"+str(i),kernel_regularizer=reg)(prev)
    
    model=Model([in_context,in_tf],prev)
    
    return model
####################################################################################################################
                                            ### GRINTON ###
####################################################################################################################
def grinnellton(grinton_params,grinnell,elton):
   meth=grinton_params["meth"]
   act=grinton_params["activ"]
   reg=get_regularizer(grinton_params.get("reg").get("regtype"),grinton_params.get("reg").get("regparam"))
   fe_ins=grinnell.input.copy()
   if(type(elton.input)==list):
       fe_ins=fe_ins+elton.input
   else:
       fe_ins.append(elton.input)
   
           
   if(meth): ###Joint
       prev=Concatenate()([grinnell.output,elton.output])
       for i in range(grinton_params.get("nl")):
           prev=Dense(grinton_params.get("nn")[i], activation=act,name="grinton_"+str(i),kernel_regularizer=reg)(prev) 
       
       out=Activation('softmax')(prev)   ###has as many outputs as targets
       
   else: ##Ensemble
       soft=[Activation('softmax')(grinnell.output),Activation('softmax')(elton.output)]  ###each one has as many outputs as targets
       out=Average()(soft)
       
   model=Model(fe_ins,out)
   
   return model
        
