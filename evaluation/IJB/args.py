class Args:
    def __init__(self, subset='IJBC', is_bunch=False, restore_embs_left=None, restore_embs_right=None, fit_mapping=False, fit_flips=False, decay_coef=0.0, pre_template_map=False, is_rotation_map=True, is_procrustes=False, explained_variance_proportion=1.0, save_result="IJB_result/{model_name}_{subset}.npz"):
        self.subset = subset
        self.is_bunch=is_bunch
        self.restore_embs_left = restore_embs_left
        self.restore_embs_right = restore_embs_right
        self.fit_mapping = fit_mapping
        self.fit_flips = fit_flips
        self.decay_coef = decay_coef
        self.pre_template_map = pre_template_map
        self.is_rotation_map = is_rotation_map
        self.is_procrustes = is_procrustes
        self.explained_variance_proportion = explained_variance_proportion
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