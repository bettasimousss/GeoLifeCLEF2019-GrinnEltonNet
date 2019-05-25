# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 17:28:27 2019

@author: saras
"""

from learning_components import cnn_comp, embcnn_comp, featTransf_merge, group_interaction_comp, embed_interaction_comp, grinnellton, tfidf_interaction_comp, emb_merg_cnn
from keras.utils.vis_utils import plot_model
from custom_metrics import focal_loss_softmax, topk_accuracy

######### Grinnellian niche learning ###########
class GrinnellianNiche(object):
    def __init__(self,sep=True,archi=0):
        self.fe_comps=None
        self.grinnell=None
        self.sep=sep
        self.archi=archi
    
    def create_grinnell(self,Anames,Pnames,clim_params, pedo_params, anthropo_params, ft_params,spat_params_list,trt,feat_names_list,im_name_list):
        if(self.sep):
            clim=cnn_comp(clim_params,"TopoHydroClim")
            anthro=embcnn_comp(anthropo_params,Anames,"LandCover")
            pedo=embcnn_comp(pedo_params,Pnames,"Pedology")
            self.fe_comps=[clim,anthro,pedo]
        else:
            self.fe_comps=[emb_merg_cnn(spat_params_list,trt,feat_names_list,im_name_list)]
        if(self.archi==0): 
            out_activ=False
        else:
            out_activ=True
        self.grinnell=featTransf_merge(ft_params,self.fe_comps,out_activ)
    
    def plot_grinnell(self,file):
        plot_model(self.grinnell,file,show_shapes=True)
    
    ######### Training functions ##########
    def compile_grinnell(self,optimizer,loss,metrics):
        self.grinnell.compile(optimizer,loss,metrics)
    
    #def train_grinnell()


class EltonianNiche(object):
    def __init__(self,final=False):
        self.elton=None
        self.final=final
    
    def create_elton(self,bio_params,emb):
        if(emb==0):
            self.elton=embed_interaction_comp(bio_params,self.final)
        elif(emb==1):
            self.elton=tfidf_interaction_comp(bio_params)
        else:
            self.elton=group_interaction_comp(bio_params)
        
    def plot_elton(self,file):
        plot_model(self.elton,file,show_shapes=True)
        
    ######### Training functions ##########
    def compile_elton(self,optimizer,loss,metrics):
        self.elton.compile(optimizer,loss,metrics)
    
    #def train_elton()

class Grinton(object):
    def __init__(self,grinnell,elton):
        self.grinnell=grinnell
        self.elton=elton
        self.grinton=None
    
    def create_grinton(self,grinton_params):
        self.grinton=grinnellton(grinton_params,self.grinnell,self.elton)
        
    def plot_grinton(self,file):
        plot_model(self.grinton,file,show_shapes=True)
        
    ######### Training functions ##########
    def compile_grinton(self,optimizer,loss,metrics):
        self.grinton.compile(optimizer,loss,metrics)
    
    #def train_grinton()


#if __name__ == '__main__':
#
#    Grinnell=GrinnellianNiche()
#    Grinnell.create_grinnell(gp.Anames,gp.Pnames,gp.topoHydroClim_params, gp.pedo_params, gp.anthropo_params, gp.ft_params,False)
#    Grinnell.plot_grinnell("grinnell_vgg_join.png")
#    Grinnell.compile_grinnell("adam",focal_loss_softmax(gamma=2),["categorical_accuracy",topk_accuracy(30)])
#    
#    Elton=EltonianNiche()
#    label=["embed","tfidf_embed","group"]
#    merge=["ensemble","joint"]
#    for m in [0,1,2]:
#        Elton.create_elton(gp.bio_params[m],m)
#        #Elton.plot_elton("elton_"+label[m]+".png")
#        cpt=0
#        Grinnellton=Grinton(Grinnell.grinnell,Elton.elton)
#        for p in [gp.ensemble_grinton_params,gp.joint_grinton_params]:
#            Grinnellton.create_grinton(p)
#            #Grinnellton.plot_grinton("grinton_"+label[m]+"_"+merge[cpt]+".png")
#            cpt+=1
