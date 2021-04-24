#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import traceback


# In[2]:


import IJB_evals as IJB


# In[3]:


from IPython.display import display


# In[4]:


class Args:
    def __init__(self, subset='IJBC', is_bunch=False, restore_embs_left=None, restore_embs_right=None, fit_mapping=False, fit_flips=False, decay_coef=0.0, pre_template_map=False, is_rotation_map=True, save_result="IJB_result/{model_name}_{subset}.npz"):
        self.subset = subset
        self.is_bunch=is_bunch
        self.restore_embs_left = restore_embs_left
        self.restore_embs_right = restore_embs_right
        self.fit_mapping = fit_mapping
        self.fit_flips = fit_flips
        self.decay_coef = decay_coef
        self.pre_template_map = pre_template_map
        self.is_rotation_map = is_rotation_map
        self.save_result = save_result
        self.save_embeddings = False
        self.model_file = None
        self.data_path = './'
        self.batch_size=64
        self.save_label=False
        self.force_reload=False
        self.is_one_2_N=False
        self.plot_only=None
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# In[5]:


dataframes = {}
fit_flips = False
decay_coef = 0.0
pre_template_map= True
is_rotation_map = True


# In[6]:


embs_list = [('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/MS1MV2-ResNet100-Arcface_IJBC.npz', 'MS1MV2', 'ResNet100', 'ArcFace'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/VGG2-ResNet50-Arcface_IJBC.npz', 'VGGFace2', 'ResNet50', 'ArcFace'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/glint360k_r100FC_0.1_IJBC.npz', 'Glint360k', 'ResNet100', 'PartialFC_r0.1'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/insightface/evaluation/IJB/IJB_result/glint360k_r100FC_1.0_IJBC.npz', 'Glint360k', 'ResNet100', 'PartialFC_r1.0'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/arcface-tf2/ijbc_embs_arc_res50.npy', 'MS1M', 'ResNet50', 'ArcFace'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/arcface-tf2/ijbc_embs_arc_mbv2.npy', 'MS1M', 'MobileNetV2', 'ArcFace'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/facenet/vggface2_ir2_ijbc_embs.npy', 'VGGFace2', 'InceptionResNetV1', 'CenterLoss'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/facenet/casia_ir2_ijbc_embs.npy', 'CASIA-WebFace', 'InceptionResNetV1', 'CenterLoss'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/Probabilistic-Face-Embeddings/ijbc_embs_pfe_sphere64_msarcface_am.npy', 'MS1M', '64-CNN', 'SphereFace+PFE'),
             ('/s/red/b/nobackup/data/portable/tbiom/models/Probabilistic-Face-Embeddings/ijbc_embs_pfe_sphere64_casia_am.npy', 'CASIA-WebFace', '64-CNN', 'SphereFace+PFE')]


# # cache maps

# In[7]:


import map_tools


# In[8]:


import importlib
importlib.reload(IJB)
importlib.reload(map_tools)


# ```
# 
# for left_embs_fn, left_dataset, left_architecture, left_head in embs_list:
#     for right_embs_fn, right_dataset, right_architecture, right_head in embs_list:
#         if left_embs_fn == right_embs_fn:
#             continue
#             
#         print(left_embs_fn, 'to', right_embs_fn)
# 
#         #for pre_template_map in pre_template_map:
#         save_result_name = 'rot-maps/{}_TO_{}.npy'.format(left_embs_fn.split('/')[-1].split('.')[0], right_embs_fn.split('/')[-1].split('.')[0])
#         if '.npz' in left_embs_fn:
#             left_embs = np.load(left_embs_fn)['embs']
#         else:
#             left_embs = np.load(left_embs_fn)
#         if '.npz' in right_embs_fn:
#             right_embs = np.load(right_embs_fn)['embs']
#         else:
#             right_embs = np.load(right_embs_fn)
#         M = map_tools.fit_rot_map(left_embs, right_embs, 11856, LR=100.0, LOG_INTERVAL=10, EPOCHS=100)
#         np.save(save_result_name, M)
# ```

# # eval maps

# In[9]:


for left_embs, left_dataset, left_architecture, left_head in embs_list:
    for right_embs, right_dataset, right_architecture, right_head in embs_list:
        if left_embs == right_embs:
            continue

        try:
            #for pre_template_map in pre_template_map:
            save_result_name = '{}_TO_{}_rotonly'.format(left_embs.split('/')[-1].split('.')[0], right_embs.split('/')[-1].split('.')[0])
            save_result = '../../../../results/{}.npz'.format(save_result_name)
            args = Args(subset='IJBC',  
                is_bunch=False,
                restore_embs_left=left_embs,
                restore_embs_right=right_embs,
                fit_mapping=True,
                fit_flips=fit_flips,
                decay_coef=decay_coef,
                pre_template_map=pre_template_map,
                is_rotation_map=is_rotation_map,
                save_result=save_result)
            df, fig = IJB.main(args)
            df['L_DATASET'] = left_dataset
            df['L_ARCH'] = left_architecture
            df['L_HEAD'] = left_head
            df['R_DATASET'] = right_dataset
            df['R_ARCH'] = right_architecture
            df['R_HEAD'] = right_head
            display(df)
            dataframes[save_result_name] = df
            print('saving to', args.save_result + '.csv')
            df.to_csv(args.save_result + '.csv')
        except Exception:
            traceback.print_exc()


# In[ ]:





# In[ ]:


import pandas as pd


# In[ ]:


superdf = pd.concat([df for df in dataframes.values()])


# In[ ]:


superdf


# In[ ]:


superdf.to_csv('../../../../results/ALL_aggregated_results_rotonly_v1.csv')


# In[ ]:




