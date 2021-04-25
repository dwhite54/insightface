#!/usr/bin/env python3
import os
import numpy as np
from tqdm.auto import tqdm
from skimage import transform
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import Ridge, LinearRegression
import pandas as pd
import cv2
import map_tools


class Mxnet_model_interf:
    def __init__(self, model_file, layer="fc1", image_size=(112, 112)):
        import mxnet as mx

        self.mx = mx
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if len(cvd) > 0 and int(cvd) != -1:
            ctx = [self.mx.gpu(ii) for ii in range(len(cvd.split(",")))]
        else:
            ctx = [self.mx.cpu()]

        prefix, epoch = model_file.split(",")
        print(">>>> loading mxnet model:", prefix, epoch, ctx)
        sym, arg_params, aux_params = self.mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers[layer + "_output"]
        model = self.mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[("data", (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        self.model = model

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2)
        data = self.mx.nd.array(imgs)
        db = self.mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()
        return emb


class Torch_model_interf:
    def __init__(self, model_file, image_size=(112, 112)):
        import torch

        self.torch = torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
        self.device = self.torch.device(device_name)
        try:
            self.model = self.torch.jit.load(model_file, map_location=device_name)
        except:
            print("Error: %s is weights only, please load and save the entire model by `torch.jit.save`" % model_file)
            self.model = None

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.transpose(0, 3, 1, 2).copy().astype("float32")
        imgs = (imgs - 127.5) * 0.0078125
        output = self.model(self.torch.from_numpy(imgs).to(self.device).float())
        return output.to('cpu').detach().numpy()


def keras_model_interf(model_file):
    import tensorflow as tf

    mm = tf.keras.models.load_model(model_file, compile=False)
    return lambda imgs: mm((tf.cast(imgs, "float32") - 127.5) * 0.0078125).numpy()


def face_align_landmark(img, landmark, image_size=(112, 112), method="similar"):
    tform = transform.AffineTransform() if method == "affine" else transform.SimilarityTransform()
    src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32
    )
    tform.estimate(landmark, src)
    # ndimage = transform.warp(img, tform.inverse, output_shape=image_size)
    # ndimage = (ndimage * 255).astype(np.uint8)
    M = tform.params[0:2, :]
    ndimage = cv2.warpAffine(img, M, image_size, borderValue=0.0)
    if len(ndimage.shape) == 2:
        ndimage = np.stack([ndimage, ndimage, ndimage], -1)
    else:
        ndimage = cv2.cvtColor(ndimage, cv2.COLOR_BGR2RGB)
    return ndimage


def read_IJB_meta_columns_to_int(file_path, columns, sep=" ", skiprows=0, header=None):
    # meta = np.loadtxt(file_path, skiprows=skiprows, delimiter=sep)
    meta = pd.read_csv(file_path, sep=sep, skiprows=skiprows, header=header).values
    return (meta[:, ii].astype("int") for ii in columns)


def extract_IJB_data_11(data_path, subset, save_path=None, force_reload=False):
    if save_path == None:
        save_path = os.path.join(data_path, subset + "_backup.npz")
    if not force_reload and os.path.exists(save_path):
        print(">>>> Reloading from backup: %s ..." % save_path)
        aa = np.load(save_path)
        return (
            aa["templates"],
            aa["medias"],
            aa["p1"],
            aa["p2"],
            aa["label"],
            aa["img_names"],
            aa["landmarks"],
            aa["face_scores"],
        )

    if subset == "IJBB":
        media_list_path = os.path.join(data_path, "IJBB/meta/ijbb_face_tid_mid.txt")
        pair_list_path = os.path.join(data_path, "IJBB/meta/ijbb_template_pair_label.txt")
        img_path = os.path.join(data_path, "IJBB/loose_crop")
        img_list_path = os.path.join(data_path, "IJBB/meta/ijbb_name_5pts_score.txt")
    else:
        media_list_path = os.path.join(data_path, "IJBC/meta/ijbc_face_tid_mid.txt")
        pair_list_path = os.path.join(data_path, "IJBC/meta/ijbc_template_pair_label.txt")
        img_path = os.path.join(data_path, "IJBC/loose_crop")
        img_list_path = os.path.join(data_path, "IJBC/meta/ijbc_name_5pts_score.txt")

    print(">>>> Loading templates and medias...")
    templates, medias = read_IJB_meta_columns_to_int(media_list_path, columns=[1, 2])  # ['1.jpg', '1', '69544']
    print("templates: %s, medias: %s, unique templates: %s" % (templates.shape, medias.shape, np.unique(templates).shape))
    # (227630,) (227630,) (12115,)

    print(">>>> Loading pairs...")
    p1, p2, label = read_IJB_meta_columns_to_int(pair_list_path, columns=[0, 1, 2])  # ['1', '11065', '1']
    print("p1: %s, unique p1: %s" % (p1.shape, np.unique(p1).shape))
    print("p2: %s, unique p2: %s" % (p2.shape, np.unique(p2).shape))
    print("label: %s, label value counts: %s" % (label.shape, dict(zip(*np.unique(label, return_counts=True)))))
    # (8010270,) (8010270,) (8010270,) (1845,) (10270,) # 10270 + 1845 = 12115
    # {0: 8000000, 1: 10270}

    print(">>>> Loading images...")
    with open(img_list_path, "r") as ff:
        # 1.jpg 46.060 62.026 87.785 60.323 68.851 77.656 52.162 99.875 86.450 98.648 0.999
        img_records = np.array([ii.strip().split(" ") for ii in ff.readlines()])

    img_names = np.array([os.path.join(img_path, ii) for ii in img_records[:, 0]])
    landmarks = img_records[:, 1:-1].astype("float32").reshape(-1, 5, 2)
    face_scores = img_records[:, -1].astype("float32")
    print("img_names: %s, landmarks: %s, face_scores: %s" % (img_names.shape, landmarks.shape, face_scores.shape))
    # (227630,) (227630, 5, 2) (227630,)
    print("face_scores value counts:", dict(zip(*np.histogram(face_scores, bins=9)[::-1])))
    # {0.1: 2515, 0.2: 0, 0.3: 62, 0.4: 94, 0.5: 136, 0.6: 197, 0.7: 291, 0.8: 538, 0.9: 223797}

    print(">>>> Saving backup to: %s ..." % save_path)
    np.savez(
        save_path,
        templates=templates,
        medias=medias,
        p1=p1,
        p2=p2,
        label=label,
        img_names=img_names,
        landmarks=landmarks,
        face_scores=face_scores,
    )
    print()
    return templates, medias, p1, p2, label, img_names, landmarks, face_scores


def extract_gallery_prob_data(data_path, subset, save_path=None, force_reload=False):
    if save_path == None:
        save_path = os.path.join(data_path, subset + "_gallery_prob_backup.npz")
    if not force_reload and os.path.exists(save_path):
        print(">>>> Reloading from backup: %s ..." % save_path)
        aa = np.load(save_path)
        return (
            aa["gallery_templates"],
            aa["gallery_subject_ids"],
            aa["probe_mixed_templates"],
            aa["probe_mixed_subject_ids"],
        )

    if subset == "IJBC":
        meta_dir = os.path.join(data_path, "IJBC/meta")
        gallery_s1_record = os.path.join(meta_dir, "ijbc_1N_gallery_G1.csv")
        gallery_s2_record = os.path.join(meta_dir, "ijbc_1N_gallery_G2.csv")
        probe_mixed_record = os.path.join(meta_dir, "ijbc_1N_probe_mixed.csv")
    else:
        meta_dir = os.path.join(data_path, "IJBB/meta")
        gallery_s1_record = os.path.join(meta_dir, "ijbb_1N_gallery_S1.csv")
        gallery_s2_record = os.path.join(meta_dir, "ijbb_1N_gallery_S2.csv")
        probe_mixed_record = os.path.join(meta_dir, "ijbb_1N_probe_mixed.csv")

    print(">>>> Loading gallery feature...")
    s1_templates, s1_subject_ids = read_IJB_meta_columns_to_int(gallery_s1_record, columns=[0, 1], skiprows=1, sep=",")
    s2_templates, s2_subject_ids = read_IJB_meta_columns_to_int(gallery_s2_record, columns=[0, 1], skiprows=1, sep=",")
    gallery_templates = np.concatenate([s1_templates, s2_templates])
    gallery_subject_ids = np.concatenate([s1_subject_ids, s2_subject_ids])
    print("s1 gallery: %s, ids: %s, unique: %s" % (s1_templates.shape, s1_subject_ids.shape, np.unique(s1_templates).shape))
    print("s2 gallery: %s, ids: %s, unique: %s" % (s2_templates.shape, s2_subject_ids.shape, np.unique(s2_templates).shape))
    print(
        "total gallery: %s, ids: %s, unique: %s"
        % (gallery_templates.shape, gallery_subject_ids.shape, np.unique(gallery_templates).shape)
    )

    print(">>>> Loading prope feature...")
    probe_mixed_templates, probe_mixed_subject_ids = read_IJB_meta_columns_to_int(
        probe_mixed_record, columns=[0, 1], skiprows=1, sep=","
    )
    print("probe_mixed_templates: %s, unique: %s" % (probe_mixed_templates.shape, np.unique(probe_mixed_templates).shape))
    print("probe_mixed_subject_ids: %s, unique: %s" % (probe_mixed_subject_ids.shape, np.unique(probe_mixed_subject_ids).shape))

    print(">>>> Saving backup to: %s ..." % save_path)
    np.savez(
        save_path,
        gallery_templates=gallery_templates,
        gallery_subject_ids=gallery_subject_ids,
        probe_mixed_templates=probe_mixed_templates,
        probe_mixed_subject_ids=probe_mixed_subject_ids,
    )
    print()
    return gallery_templates, gallery_subject_ids, probe_mixed_templates, probe_mixed_subject_ids


def get_embeddings(model_interf, img_names, landmarks, batch_size=64, flip=True, M=None, print_log=True):
    steps = int(np.ceil(len(img_names) / batch_size))
    embs, embs_f = [], []
    batches = range(0, len(img_names), batch_size)
    if print_log:
        batches = tqdm(batches, "Embedding", total=steps)
    for batch_id in batches:
        batch_imgs, batch_landmarks = img_names[batch_id : batch_id + batch_size], landmarks[batch_id : batch_id + batch_size]
        ndimages = [face_align_landmark(cv2.imread(img), landmark) for img, landmark in zip(batch_imgs, batch_landmarks)]
        ndimages = np.stack(ndimages)
        embs.extend(model_interf(ndimages))
        if flip:
            embs_f.extend(model_interf(ndimages[:, :, ::-1, :]))
            
    return np.array(embs), np.array(embs_f)


def process_embeddings(embs, embs_f=[], use_flip_test=True, use_norm_score=False, use_detector_score=True, face_scores=None):
    if use_flip_test and len(embs_f) != 0:
        embs = embs + embs_f
    if use_norm_score:
        embs = normalize(embs)
    if use_detector_score and face_scores is not None:
        embs = embs * np.expand_dims(face_scores, -1)
    return embs


def image2template_feature(img_feats=None, templates=None, medias=None, choose_templates=None, choose_ids=None, print_log=True):
    if choose_templates is not None:  # 1N
        unique_templates, indices = np.unique(choose_templates, return_index=True)
        unique_subjectids = choose_ids[indices]
    else:  # 11
        unique_templates = np.unique(templates)
        unique_subjectids = None

    # template_feats = np.zeros((len(unique_templates), img_feats.shape[1]), dtype=img_feats.dtype)
    template_feats = np.zeros((len(unique_templates), img_feats.shape[1]))
    uqts = enumerate(unique_templates)
    if print_log:
        uqts = tqdm(uqts, "Extract template feature", total=len(unique_templates))
    for count_template, uqt in uqts:
        (ind_t,) = np.where(templates == uqt)
        face_norm_feats = img_feats[ind_t]
        face_medias = medias[ind_t]
        unique_medias, unique_media_counts = np.unique(face_medias, return_counts=True)
        media_norm_feats = []
        for u, ct in zip(unique_medias, unique_media_counts):
            (ind_m,) = np.where(face_medias == u)
            if ct == 1:
                media_norm_feats += [face_norm_feats[ind_m]]
            else:  # image features from the same video will be aggregated into one feature
                media_norm_feats += [np.mean(face_norm_feats[ind_m], 0, keepdims=True)]
        media_norm_feats = np.array(media_norm_feats)
        # media_norm_feats = media_norm_feats / np.sqrt(np.sum(media_norm_feats ** 2, -1, keepdims=True))
        template_feats[count_template] = np.sum(media_norm_feats, 0)
    template_norm_feats = normalize(template_feats)
    return template_norm_feats, unique_templates, unique_subjectids


def verification_11(template_norm_feats=None, unique_templates=None, p1=None, p2=None, template_norm_feats_right=None, batch_size=100000, print_log=True):
    template2id = np.zeros(max(unique_templates) + 1, dtype=int)
    for count_template, uqt in enumerate(unique_templates):
        template2id[uqt] = count_template        

    steps = int(np.ceil(len(p1) / batch_size))
    score = []
    ids = range(steps)
    if print_log:
        ids = tqdm(ids, "Verification")
    for id in ids:
        feat1 = template_norm_feats[template2id[p1[id * batch_size : (id + 1) * batch_size]]]
        if template_norm_feats_right is not None:
            feat2 = template_norm_feats_right[template2id[p2[id * batch_size : (id + 1) * batch_size]]]
        else:
            feat2 = template_norm_feats[template2id[p2[id * batch_size : (id + 1) * batch_size]]]
        score.extend(np.sum(feat1 * feat2, -1))
    return np.array(score)


def evaluation_1N(query_feats, gallery_feats, query_ids, reg_ids):
    import heapq

    Fars = [0.01, 0.1]
    print("query_feats: %s, gallery_feats: %s" % (query_feats.shape, gallery_feats.shape))

    query_num = query_feats.shape[0]
    gallery_num = gallery_feats.shape[0]

    similarity = np.dot(query_feats, gallery_feats.T)
    print("similarity shape:", similarity.shape)
    top_inds = np.argsort(-similarity)
    print("top_inds shape:", top_inds.shape)

    # gen_mask
    mask = []
    for query_id in query_ids:
        pos = [i for i, x in enumerate(reg_ids) if query_id == x]
        if len(pos) != 1:
            raise RuntimeError("RegIdsError with id = {}， duplicate = {} ".format(query_id, len(pos)))
        mask.append(pos[0])

    # calculate top_n
    correct_num_1, correct_num_5, correct_num_10 = 0, 0, 0
    for i in range(query_num):
        top_1, top_5, top_10 = top_inds[i, 0], top_inds[i, 0:5], top_inds[i, 0:10]
        if mask[i] == top_1:
            correct_num_1 += 1
        if mask[i] in top_5:
            correct_num_5 += 1
        if mask[i] in top_10:
            correct_num_10 += 1
    print("top1: %f, top5: %f, top10: %f" % (correct_num_1 / query_num, correct_num_5 / query_num, correct_num_10 / query_num))

    # neg_pair_num = query_num * gallery_num - query_num
    # print("neg_pair_num:", neg_pair_num)
    required_topk = [int(np.ceil(query_num * x)) for x in Fars]
    top_sims = similarity
    # calculate fars and tprs
    pos_sims = []
    for i in range(query_num):
        gt = mask[i]
        pos_sims.append(top_sims[i, gt])
        top_sims[i, gt] = -2.0

    pos_sims = np.array(pos_sims)
    neg_sims = top_sims[np.where(top_sims > -2.0)]
    neg_sims_sorted = heapq.nlargest(max(required_topk), neg_sims)  # heap sort
    print("pos_sims: %s, neg_sims: %s, neg_sims_sorted: %d" % (pos_sims.shape, neg_sims.shape, len(neg_sims_sorted)))
    for far, pos in zip(Fars, required_topk):
        th = neg_sims_sorted[pos - 1]
        recall = np.sum(pos_sims > th) / query_num
        print("far = {:.10f} pr = {:.10f} th = {:.10f}".format(far, recall, th))


class IJB_test:
    def __init__(self, args):
        templates, medias, p1, p2, label, img_names, landmarks, face_scores = extract_IJB_data_11(
            args.data_path, args.subset, force_reload=args.force_reload
        )
        self.print_log = args.print_log
        
        self.embs_right = None
        self.embs_f_right = None
        self.mapping = None
        
        if args.model_file != None:
            if args.model_file.endswith(".h5"):
                interf_func = keras_model_interf(args.model_file)
            elif args.model_file.endswith(".pth") or args.model_file.endswith(".pt"):
                interf_func = Torch_model_interf(args.model_file)
            else:
                interf_func = Mxnet_model_interf(args.model_file)
            self.embs, self.embs_f = get_embeddings(interf_func, img_names, landmarks, batch_size=args.batch_size, print_log=self.print_log)
        elif args.restore_embs_left is not None:
            print(">>>> Reload (left) embeddings from:", args.restore_embs_left)
            aa = np.load(args.restore_embs_left)
            if '.npz' in args.restore_embs_left and "embs" in aa and "embs_f" in aa:
                self.embs, self.embs_f = aa["embs"], aa["embs_f"]
            elif '.npy' in args.restore_embs_left:
                self.embs, self.embs_f = aa, aa
            else:
                print("ERROR: %s NOT containing embs / embs_f" % args.restore_embs_left)
                exit(1)
            if args.restore_embs_right is not None:
                print(">>>> Reload (right) embeddings from:", args.restore_embs_right)
                bb = np.load(args.restore_embs_right)
                if '.npz' in args.restore_embs_right and "embs" in bb and "embs_f" in bb:
                    self.embs_right, self.embs_f_right = bb["embs"], bb["embs_f"]
                elif '.npy' in args.restore_embs_right:
                    self.embs_right, self.embs_f_right = bb, bb
                else:
                    print("ERROR: %s NOT containing embs / embs_f" % args.restore_embs_right)
                    exit(1)
                
                if args.fit_mapping:
                    print(">>>> fit {} mapping with {} individuals ({})".format(('rotation' if args.is_rotation_map else 'ridge'), args.n_individuals, args.map_normed))
                    train_idx = np.arange(11856)  # all enroll individuals
                    if args.n_individuals != -1:
                        train_idx = np.random.choice(train_idx, args.n_individuals, replace=False)
                    train_left = self.embs[train_idx]
                    train_right = self.embs_right[train_idx]
                    # TODO ensure all embeddings have flips--most do not
#                     if fit_flips:
#                         enroll_left = self.embs[train_idx] + self.embs_f[train_idx]
#                         enroll_right = self.embs_right[train_idx] + self.embs_f_right[train_idx]
#                     else:
#                         enroll_left = self.embs[train_idx]
#                         enroll_right = self.embs_right[train_idx]
                    if args.map_normed:
                        train_left = normalize(train_left)
                        train_right = normalize(train_right)
                    if args.is_rotation_map:
                        self.mapping = map_tools.fit_procrustes_map(train_left, train_right, is_wahba=True)
                    else:
                        self.mapping = map_tools.fit_map(train_left, train_right, args.decay_coef)
            print(">>>> Done loading, begin verification.")
        self.data_path, self.subset, self.force_reload = args.data_path, args.subset, args.force_reload
        self.templates, self.medias, self.p1, self.p2, self.label = templates, medias, p1, p2, label
        self.face_scores = face_scores.astype(self.embs.dtype)

    def run_model_test_single(self, use_flip_test=True, use_norm_score=False, use_detector_score=True, fit_mapping=False):
        if fit_mapping:
            #embs = self.mapping.predict(self.embs)
            embs = self.embs @ self.mapping
            #embs_f = self.mapping.predict(self.embs_f)
            #embs_f = self.embs_f @ self.mapping
        else:
            embs = self.embs
            #embs_f = self.embs_f
        img_input_feats = process_embeddings(
                embs, #embs_f,
                use_flip_test=use_flip_test,
                use_norm_score=use_norm_score,
                use_detector_score=use_detector_score,
                face_scores=self.face_scores,
        )
        template_norm_feats, unique_templates, _ = image2template_feature(img_input_feats, self.templates, self.medias, print_log=self.print_log)
        if self.embs_right is not None:# and self.embs_f_right is not None:
            img_input_feats_right = process_embeddings(
                self.embs_right, #self.embs_f_right,
                use_flip_test=use_flip_test,
                use_norm_score=use_norm_score,
                use_detector_score=use_detector_score,
                face_scores=self.face_scores,
            )
            template_norm_feats_right, _, _ = image2template_feature(img_input_feats_right, self.templates, self.medias, print_log=self.print_log)
            score = verification_11(template_norm_feats, unique_templates, self.p1, self.p2, template_norm_feats_right, print_log=self.print_log)
        else:
            score = verification_11(template_norm_feats, unique_templates, self.p1, self.p2, print_log=self.print_log)
        return score

    def run_model_test_bunch(self, fit_mapping=False):
        from itertools import product

        scores, names = [], []
        for use_norm_score, use_detector_score, use_flip_test in product([True, False], [True, False], [True, False]):
            name = "N{:d}D{:d}F{:d}".format(use_norm_score, use_detector_score, use_flip_test)
            print(">>>>", name, use_norm_score, use_detector_score, use_flip_test)
            names.append(name)
            scores.append(self.run_model_test_single(use_flip_test, use_norm_score, use_detector_score, fit_mapping))
        return scores, names

    def run_model_test_1N(self):
        gallery_templates, gallery_subject_ids, probe_mixed_templates, probe_mixed_subject_ids = extract_gallery_prob_data(
            self.data_path, self.subset, force_reload=self.force_reload
        )
        img_input_feats = process_embeddings(
            self.embs,
            self.embs_f,
            use_flip_test=True,
            use_norm_score=False,
            use_detector_score=True,
            face_scores=self.face_scores,
        )
        gallery_templates_feature, gallery_unique_templates, gallery_unique_subject_ids = image2template_feature(
            img_input_feats, self.templates, self.medias, gallery_templates, gallery_subject_ids, print_log=self.print_log
        )
        print("gallery_templates_feature:", gallery_templates_feature.shape)
        print("gallery_unique_subject_ids:", gallery_unique_subject_ids.shape)

        probe_mixed_templates_feature, probe_mixed_unique_templates, probe_mixed_unique_subject_ids = image2template_feature(
            img_input_feats, self.templates, self.medias, probe_mixed_templates, probe_mixed_subject_ids, print_log=self.print_log
        )
        print("probe_mixed_templates_feature:", probe_mixed_templates_feature.shape)
        print("probe_mixed_unique_subject_ids:", probe_mixed_unique_subject_ids.shape)

        evaluation_1N(
            probe_mixed_templates_feature, gallery_templates_feature, probe_mixed_unique_subject_ids, gallery_unique_subject_ids
        )


def plot_roc_and_calculate_tpr(scores, names=None, label=None):
    print(">>>> plot roc and calculate tpr...")
    score_dict = {}
    for id, score in enumerate(scores):
        name = None if names is None else names[id]
        if isinstance(score, str) and score.endswith(".npz"):
            aa = np.load(score)
            score = aa.get("scores", [])
            label = aa["label"] if label is None and "label" in aa else label
            score_name = aa.get("names", [])
            for ss, nn in zip(score, score_name):
                score_dict[nn] = ss
        elif isinstance(score, str) and score.endswith(".npy"):
            name = name if name is not None else os.path.splitext(os.path.basename(score))[0]
            score_dict[name] = np.load(score)
        elif isinstance(score, str) and score.endswith(".txt"):
            # IJB meta data like ijbb_template_pair_label.txt
            label = pd.read_csv(score, sep=" ").values[:, 2]
        else:
            name = name if name is not None else str(id)
            score_dict[name] = score
    if label is None:
        print("Error: Label data is not provided")
        return None, None

    x_labels = [10 ** (-ii) for ii in range(1, 7)[::-1]]
    fpr_dict, tpr_dict, roc_auc_dict, tpr_result = {}, {}, {}, {}
    for name, score in score_dict.items():
        fpr, tpr, _ = roc_curve(label, score)
        roc_auc = auc(fpr, tpr)
        fpr, tpr = np.flipud(fpr), np.flipud(tpr)  # select largest tpr at same fpr
        tpr_result[name] = [tpr[np.argmin(abs(fpr - ii))] for ii in x_labels]
        fpr_dict[name], tpr_dict[name], roc_auc_dict[name] = fpr, tpr, roc_auc
    tpr_result_df = pd.DataFrame(tpr_result, index=x_labels).T
    tpr_result_df.columns.name = "Methods"
    #print(tpr_result_df.to_markdown())
    #print(tpr_result_df)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure()
        for name in score_dict:
            plt.plot(fpr_dict[name], tpr_dict[name], lw=1, label="[%s (AUC = %0.4f%%)]" % (name, roc_auc_dict[name] * 100))
            print("[%s (AUC = %0.4f%%)]" % (name, roc_auc_dict[name] * 100))

        plt.xlim([10 ** -6, 0.1])
        plt.ylim([0.3, 1.0])
        plt.grid(linestyle="--", linewidth=1)
        plt.xticks(x_labels)
        plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
        plt.xscale("log")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC on IJB")
        #plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()
    except:
        print("Missing matplotlib")
        fig = None

    return tpr_result_df, fig


def parse_arguments(argv):
    import argparse

    default_save_result_name = "IJB_result/{model_name}_{subset}.npz"
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model_file", type=str, default=None, help="Saved model file, could be keras h5 / pytorch jit pth / mxnet")
    parser.add_argument("-d", "--data_path", type=str, default="./", help="Dataset path")
    parser.add_argument("-s", "--subset", type=str, default="IJBB", help="Subset test target, could be IJBB / IJBC")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="Batch size for get_embeddings")
    parser.add_argument("-n", "--n_individuals", type=int, default=-1, help="Number of individuals for fitting map")
    parser.add_argument("-c", "--decay_coef", type=float, default=0.0, help="Weight decay coefficient for mapping fitting")
    parser.add_argument("-v", "--explained_variance_proportion", type=float, default=0.0, help="Explained variance for procrustes singular value thresholding")
    parser.add_argument("-r", "--restore_embs_left", type=str, default=default_save_result_name, help="path to result npz containing key 'embs' and 'embs_f'")
    parser.add_argument("-q", "--restore_embs_right", type=str, default=default_save_result_name, help="path to result npz containing key 'embs' and 'embs_f'")
    parser.add_argument(
        "-S", "--save_result", type=str, default=default_save_result_name, help="Filename for saving / restore result"
    )
    parser.add_argument("-L", "--save_label", action="store_true", help="Save label data, useful for plot only")
    parser.add_argument("-E", "--save_embeddings", action="store_true", help="Save embeddings data")
    parser.add_argument("-B", "--is_bunch", action="store_true", help="Run all 8 tests N{0,1}D{0,1}F{0,1}")
    parser.add_argument("-N", "--is_one_2_N", action="store_true", help="Run 1:N test instead of 1:1")
    parser.add_argument("-D", "--use_face_scores", action="store_true", help="Use face detector scores during template generation")
    parser.add_argument("-U", "--use_norm_scores", action="store_true", help="Use normed features during template generation")
    parser.add_argument("-F", "--force_reload", action="store_true", help="Force reload, instead of using cache")
    parser.add_argument("-M", "--fit_mapping", action="store_true", help="Fit mapping between left and right embeddings (-r and -q)")
    parser.add_argument("-Y", "--fit_flips", action="store_true", help="Fit mapping using flipped embeddings also")
    parser.add_argument("-T", "--pre_template_map", action="store_true", help="Map before generating templates (false for after)")
    parser.add_argument("-Q", "--map_normed", action="store_true", help="Map between normed features")
    parser.add_argument("-R", "--is_rotation_map", action="store_true", help="Fit rotation-only map (using SGD)")
    #parser.add_argument("-C", "--is_procrustes", action="store_true", help="Fit orthonormal map (using procrustes)")
    parser.add_argument("-V", "--print_log", action="store_true", help="Print progress bars (omit to only print standard messages)")
    parser.add_argument("-P", "--plot_only", nargs="*", type=str, help="Plot saved results, Format 1 2 3 or 1, 2, 3 or *.npy")
    args = parser.parse_known_args(argv)[0]
    print('args', args)

    if args.plot_only != None and len(args.plot_only) != 0:
        # Plot only
        from glob2 import glob

        score_files = []
        for ss in args.plot_only:
            score_files.extend(glob(ss.replace(",", "").strip()))
        args.plot_only = score_files
    elif (args.restore_embs_left is None or args.restore_embs_right is None) and (args.model_file == None and args.save_result == default_save_result_name):
        print("Please provide -m MODEL_FILE, or restore embeddings, see `--help` for usage.")
        exit(1)
    elif args.model_file != None:
        if args.model_file.endswith(".h5") or args.model_file.endswith(".pth") or args.model_file.endswith(".pt"):
            # Keras model file "model.h5", pytorch model ends with `.pth` or `.pt`
            model_name = os.path.splitext(os.path.basename(args.model_file))[0]
        else:
            # MXNet model file "models/r50-arcface-emore/model,1"
            model_name = os.path.basename(os.path.dirname(args.model_file))

        if args.save_result == default_save_result_name:
            args.save_result = default_save_result_name.format(model_name=model_name, subset=args.subset)
    return args


def main(args):
    #print(args)
    save_name = os.path.splitext(args.save_result)[0]
    save_items = {}
    tt = IJB_test(args)
    if args.is_one_2_N:  # 1:N test
        tt.run_model_test_1N(fit_mapping=args.fit_mapping)
    elif args.is_bunch:  # All 8 tests N{0,1}D{0,1}F{0,1}
        scores, names = tt.run_model_test_bunch(fit_mapping=args.fit_mapping)
        names = [save_name + "_" + ii for ii in names]
        save_items.update({"scores": scores, "names": names})
    else:  # Basic 1:1 N0D1F1 test
        score = tt.run_model_test_single(use_flip_test=args.fit_flips, use_norm_score=args.use_norm_scores, use_detector_score=args.use_face_scores, fit_mapping=args.fit_mapping)
        scores, names = [score], [save_name]
        save_items.update({"scores": scores, "names": names})

    if args.save_embeddings:
        save_items.update({"embs": tt.embs, "embs_f": tt.embs_f})
    if args.save_label:
        save_items.update({"label": tt.label})
        
    if args.fit_mapping:
        save_items.update({"mapping": tt.mapping})

    df, fig = None, None
    if not args.is_one_2_N:
        df, fig = plot_roc_and_calculate_tpr(scores, names=names, label=tt.label)
        save_items.update({"FAR":df.columns[:7].astype(np.float).to_numpy().astype(np.float)})
        save_items.update({"TAR@FAR":df.to_numpy()[0, :7].astype(np.float)})

    if args.model_file != None or args.save_embeddings or args.fit_mapping or not args.is_one_2_N:  # embeddings not restored from file or should save_embeddings again
        save_path = os.path.dirname(args.save_result)
        if save_path != '' and not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez(args.save_result, **save_items)
        
    return df, fig
    

if __name__ == "__main__":
    
    import sys

    args = parse_arguments(sys.argv[1:])
    df, fig = main(args)
    if df is not None and fig is not None:
        print(df)
        fig.show()
    
#     if args.plot_only != None and len(args.plot_only) != 0:
#         plot_roc_and_calculate_tpr(args.plot_only)
#     else:
#         save_name = os.path.splitext(args.save_result)[0]
#         save_items = {}
#         tt = IJB_test(args.model_file, 
#                       args.data_path, 
#                       args.subset, 
#                       args.batch_size, 
#                       args.force_reload, 
#                       args.restore_embs_left, 
#                       args.restore_embs_right, 
#                       fit_mapping=args.fit_mapping, 
#                       decay_coef=args.decay_coef, 
#                       fit_flips=args.fit_flips,
#                       )
#         if args.is_one_2_N:  # 1:N test
#             tt.run_model_test_1N(pre_template_map=args.pre_template_map)
#         elif args.is_bunch:  # All 8 tests N{0,1}D{0,1}F{0,1}
#             scores, names = tt.run_model_test_bunch(pre_template_map=args.pre_template_map)
#             names = [save_name + "_" + ii for ii in names]
#             save_items.update({"scores": scores, "names": names})
#         else:  # Basic 1:1 N0D1F1 test
#             score = tt.run_model_test_single()
#             scores, names = [score], [save_name]
#             save_items.update({"scores": scores, "names": names})

#         if args.save_embeddings:
#             save_items.update({"embs": tt.embs, "embs_f": tt.embs_f})
#         if args.save_label:
#             save_items.update({"label": tt.label})

#         if args.model_file != None or args.save_embeddings:  # embeddings not restored from file or should save_embeddings again
#             save_path = os.path.dirname(args.save_result)
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#             np.savez(args.save_result, **save_items)

#         if not args.is_one_2_N:
#             df, fig = plot_roc_and_calculate_tpr(scores, names=names, label=tt.label)
            
#             return df, fig
#         else:
#             return None, None
