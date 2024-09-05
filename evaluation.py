"""Evaluation"""

from __future__ import print_function
import os
import sys
from tkinter import Variable
import torch
import numpy as np
from utils import cosine_similarity_matrix

from model.CRCL import CRCL
from data import get_test_loader
from vocab import deserialize_vocab
from collections import OrderedDict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print, sub=0):
    """Encode all images and captions loadable by `data_loader`
    """
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None

    max_n_word = 0
    for i, (images, _, captions, lengths, ids) in enumerate(data_loader):
        lengths = lengths.numpy().astype(np.int64).tolist()
        max_n_word = max(max_n_word, max(lengths))
    ids_ = []
    for i, (images, _, captions, lengths, ids) in enumerate(data_loader):
        lengths = lengths.numpy().astype(np.int64).tolist()
        # make sure val logger is used
        model.logger = val_logger
        ids_ += ids
        # compute the embeddings
        with torch.no_grad():
            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)
        if img_embs is None:
            # img_embs = np.zeros((100 * 100, img_emb.size(1), img_emb.size(2)))
            # cap_embs = np.zeros((100 * 100, max_n_word, cap_emb.size(2)))
            # cap_lens = [0] * (100 * 100)
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids, :, :] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        del images, captions
        if sub > 0:
            print(f"===>batch {i}")
        if sub > 0 and i > sub:
            break
    if sub > 0:
        return np.array(img_embs)[ids_].tolist(), np.array(cap_embs)[ids_].tolist(), np.array(cap_lens)[
            ids_].tolist(), ids_
    else:
        return img_embs, cap_embs, cap_lens


def shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100):
    n_im_shard = (len(img_embs) - 1) // shard_size + 1
    n_cap_shard = (len(cap_embs) - 1) // shard_size + 1

    sims = np.zeros((len(img_embs), len(cap_embs)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(img_embs))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_attn_scores batch (%d,%d)' % (i, j))
            ca_start, ca_end = shard_size * j, min(shard_size * (j + 1), len(cap_embs))

            with torch.no_grad():
                im = torch.from_numpy(img_embs[im_start:im_end]).float().cuda()
                ca = torch.from_numpy(cap_embs[ca_start:ca_end]).float().cuda()
                l = cap_lens[ca_start:ca_end]
                sim, _, _, _ = model.forward_sim(im, ca, l)

            sims[im_start:im_end, ca_start:ca_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return sims


def shard_xattn_t2i(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_t2i(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_i2t(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def t2i(npts, sims, per_captions=1, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(per_captions * npts)
    top1 = np.zeros(per_captions * npts)
    top5 = np.zeros((per_captions * npts, 5), dtype=int)

    # --> (per_captions * N(caption), N(image))
    sims = sims.T
    retreivaled_index = []
    for index in range(npts):
        for i in range(per_captions):
            inds = np.argsort(sims[per_captions * index + i])[::-1]
            retreivaled_index.append(inds)
            ranks[per_captions * index + i] = np.where(inds == index)[0][0]
            top1[per_captions * index + i] = inds[0]
            top5[per_captions * index + i] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def i2t(npts, sims, per_captions=1, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (per_captions * N, max_n_word, d) matrix of captions
    CapLens: (per_captions * N) array of caption lengths
    sims: (N, per_captions * N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    top5 = np.zeros((npts, 5), dtype=int)
    retreivaled_index = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        retreivaled_index.append(inds)
        # Score
        rank = 1e20
        for i in range(per_captions * index, per_captions * index + per_captions, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        top5[index] = inds[0:5]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1, top5, retreivaled_index)
    else:
        return (r1, r5, r10, medr, meanr)


def encode_data_vse(model, data_loader):
    """Encode all images and captions loadable by `data_loader`
    """
    model.val_start()
    img_embs = None
    cap_embs = None
    for i, data_i in enumerate(data_loader):
        images, image_lengths, captions, caption_lengths, ids = data_i
        # print(images.size(),captions.size(),caption_lengths)
        img_emb, cap_emb = model.forward_emb(images, captions, caption_lengths, image_lengths)
        if img_embs is None:
            img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
        # cache embeddings
        img_embs[ids, :] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids, :] = cap_emb.data.cpu().numpy().copy()
        del images, captions

    return img_embs, cap_embs


def i2t_vse(npts, sims, return_ranks=False, mode='coco', per=5):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(per * index, per * index + per, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i_vse(npts, sims, return_ranks=False, mode='coco', per=5):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(per * npts)
        top1 = np.zeros(per * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(per):
                inds = np.argsort(sims[per * index + i])[::-1]
                ranks[per * index + i] = np.where(inds == index)[0][0]
                top1[per * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def validation_SGR_or_SAF(opt, val_loader, model, fold=False):
    # compute the encoding for all the validation images and captions
    if opt.data_name == 'cc152k_precomp':
        per_captions = 1
    elif opt.data_name in ['coco_precomp', 'f30k_precomp']:
        per_captions = 5
    else:
        print(f"No dataset")
        return 0

    model.val_start()
    print('Encoding with model')
    img_embs, cap_embs, cap_lens = encode_data(model.base_model, val_loader)
    # clear duplicate 5*images and keep 1*images FIXME
    if not fold:
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        # record computation time of validation
        print('Computing similarity from model')
        sims_mean = shard_attn_scores(model.base_model, img_embs, cap_embs, cap_lens, opt, shard_size=1000)
        print("Calculate similarity time with model")
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
        print("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
        print("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
        r_sum = r1 + r5 + r10 + r1i + r5i + r10i
        print("Sum of Recall: %.2f" % (r_sum))
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            sims = shard_attn_scores(model.base_model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt,
                                     shard_size=1000,
                                     )

            print('Computing similarity from model')
            r, rt = i2t(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)
            ri, rti = t2i(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)

            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]
        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        a = np.array(mean_metrics)
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])
        print("rsum: %.1f" % (a[0:3].sum() + a[5:8].sum()))


def validation_SGRAF(opt, val_loader, models, fold=False):
    # compute the encoding for all the validation images and captions
    if opt.data_name == 'cc152k_precomp':
        per_captions = 1
    elif opt.data_name in ['coco_precomp', 'f30k_precomp']:
        per_captions = 5
    else:
        print(f"No dataset")
        return 0

    models[0].val_start()
    models[1].val_start()
    print('Encoding with model')
    img_embs, cap_embs, cap_lens = encode_data(models[0].base_model, val_loader, opt.log_step)
    img_embs1, cap_embs1, cap_lens1 = encode_data(models[1].base_model, val_loader, opt.log_step)
    if not fold:
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        img_embs1 = np.array([img_embs1[i] for i in range(0, len(img_embs1), per_captions)])
        # record computation time of validation
        print('Computing similarity from model')
        sims_mean = shard_attn_scores(models[0].base_model, img_embs, cap_embs, cap_lens, opt, shard_size=1000,
                                      )
        sims_mean += shard_attn_scores(models[1].base_model, img_embs1, cap_embs1, cap_lens1, opt,
                                       shard_size=1000, )
        sims_mean /= 2

        # np.save("./sims/f30k/sim.npy", sims_mean)

        print("Calculate similarity time with model")
        # caption retrieval
        (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average i2t Recall: %.2f" % ((r1 + r5 + r10) / 3))
        print("Image to text: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1, r5, r10, medr, meanr))
        # image retrieval
        (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions, return_ranks=False)
        print("Average t2i Recall: %.2f" % ((r1i + r5i + r10i) / 3))
        print("Text to image: %.2f, %.2f, %.2f, %.2f, %.2f" % (r1i, r5i, r10i, medri, meanr))
        r_sum = r1 + r5 + r10 + r1i + r5i + r10i
        print("Sum of Recall: %.2f" % (r_sum))
        return r_sum
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]

            img_embs_shard1 = img_embs1[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard1 = cap_embs1[i * 5000:(i + 1) * 5000]
            cap_lens_shard1 = cap_lens1[i * 5000:(i + 1) * 5000]
            sims = shard_attn_scores(models[0].base_model, img_embs_shard, cap_embs_shard, cap_lens_shard, opt,
                                     shard_size=1000,
                                     )
            sims += shard_attn_scores(models[1].base_model, img_embs_shard1, cap_embs_shard1, cap_lens_shard1,
                                      opt,
                                      shard_size=1000,
                                      )
            sims /= 2

            print('Computing similarity from model')
            r, rt0 = i2t(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)
            ri, rti0 = t2i(img_embs_shard.shape[0], sims, per_captions, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)
            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        a = np.array(mean_metrics)

        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])
        print("rsum: %.1f" % (a[0:3].sum() + a[5:8].sum()))


def validation_VSEinfty(opt, val_loader, model, fold=False):
    # compute the encoding for all the validation images and captions
    if opt.data_name in ['cc152k_precomp', 'cc510k_precomp']:
        per_captions = 1
    elif opt.data_name in ['coco_precomp', 'f30k_precomp']:
        per_captions = 5

    model.val_start()
    if not fold:
        with torch.no_grad():
            img_embs, cap_embs = encode_data_vse(model.base_model, val_loader)
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
            sims = cosine_similarity_matrix(img_embs, cap_embs)
            npts = img_embs.shape[0]
            # np.save('./sims/f30K_0.4_RSRL.npy',sims)
            (r1, r5, r10, medr, meanr) = i2t_vse(npts, sims, per=per_captions)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                  (r1, r5, r10, medr, meanr))
            (r1i, r5i, r10i, medri, meanr) = t2i_vse(npts, sims, per=per_captions)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                  (r1i, r5i, r10i, medri, meanr))
            r_sum = r1 + r5 + r10 + r1i + r5i + r10i
            print('Current rsum is {}'.format(r_sum))
            return r1, r5, r10, r1i, r5i, r10i
    else:
        with torch.no_grad():
            results = []
            img_embs, cap_embs = encode_data_vse(model.base_model, val_loader)
            for i in range(5):
                print("fold: {}".format(i + 1))
                img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                sims = cosine_similarity_matrix(img_embs_shard, cap_embs_shard)
                npts = img_embs_shard.shape[0]
                (r1, r5, r10, medr, meanr) = i2t_vse(npts, sims, per=per_captions)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                      (r1, r5, r10, medr, meanr))
                (r1i, r5i, r10i, medri, meanr) = t2i_vse(npts, sims, per=per_captions)
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                      (r1i, r5i, r10i, medri, meanr))
                r_sum = r1 + r5 + r10 + r1i + r5i + r10i
                print('Current rsum is {}'.format(r_sum))
                results.append([r1, r5, r10, medr, meanr, r1i, r5i, r10i, medri, meanr, r_sum])

            print("-----------------------------------")
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            print("rsum: %.1f" % (mean_metrics[10]))
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[:5])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[5:10])
