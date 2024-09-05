"""Dataloader"""

import csv
import torch
import torch.utils.data as data
import os
import nltk
import numpy as np
import random
import h5py


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp, cc152k_precomp
    """

    def __init__(self, data_path, data_split, vocab, opt=None, logger=None):
        self.vocab = vocab
        self.data_split = data_split
        loc = data_path + '/'
        self.module = opt.module_name

        # load the raw captions
        self.captions = []

        if 'cc152k' in opt.data_name:
            with open(loc + '%s_caps.tsv' % data_split) as f:
                tsvreader = csv.reader(f, delimiter='\t')
                for line in tsvreader:
                    self.captions.append(line[1].strip())
        else:
            with open(loc + '%s_caps.txt' % data_split, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    self.captions.append(line.strip())

        self.images = np.load(loc + '%s_ims.npy' % data_split)

        # rkiros data has redundancy in images, we divide by 5
        img_len = self.images.shape[0]
        self.length = len(self.captions)
        if img_len != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == 'dev':
            self.length = 5000
            if 'cc152k' in opt.data_name:
                self.length = 1000

        self.noise_type = 'NCR'
        if self.noise_type == 'DECL':
            self.noisy_inx = np.arange(img_len)
            if data_split == 'train' and opt.noise_ratio > 0.0:
                noise_file = opt.noise_file
                if os.path.exists(noise_file):
                    logger.info('=> load noisy index from {}'.format(noise_file))
                    self.noisy_inx = np.load(noise_file)
                else:
                    noisy_ratio = opt.noise_ratio
                    inx = np.arange(img_len)
                    np.random.shuffle(inx)
                    noisy_inx = inx[0: int(noisy_ratio * img_len)]
                    shuffle_noisy_inx = np.array(noisy_inx)
                    np.random.shuffle(shuffle_noisy_inx)
                    self.noisy_inx[noisy_inx] = shuffle_noisy_inx
                    np.save(noise_file, self.noisy_inx)
                    logger.info('Noisy rate: %g' % noisy_ratio)

            self.real_correspondence = np.zeros(self.length)
            for i in range(self.length):
                if self.noisy_inx[i // self.im_div] == i // 5:
                    self.real_correspondence[i] = 1.0
        else:
            self.noisy_inx = np.arange(0, self.length) // self.im_div
            if data_split == 'train' and opt.noise_ratio > 0.0:
                noise_file = opt.noise_file
                if os.path.exists(noise_file):
                    logger.info('=> load noisy index from {}'.format(noise_file))
                    self.noisy_inx = np.load(noise_file)
                else:
                    idx = np.arange(self.length)
                    np.random.shuffle(idx)
                    noise_length = int(opt.noise_ratio * self.length)
                    shuffle_index = self.noisy_inx[idx[:noise_length]]
                    np.random.shuffle(shuffle_index)
                    self.noisy_inx[idx[:noise_length]] = shuffle_index
                    np.save(noise_file, self.noisy_inx)
                    logger.info("=> save noisy index to {}".format(noise_file))

            self.real_correspondence = np.zeros(self.length)
            for i in range(self.length):
                if self.noisy_inx[i] == i // 5:
                    self.real_correspondence[i] = 1.0
            print(data_split, self.noisy_inx.shape, self.real_correspondence.sum())

    def process_caption(self, caption, caption_enhance):
        enhance = caption_enhance if self.data_split == 'train' else False
        if not enhance:
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            caption = list()
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            target = torch.Tensor(caption)
            return target
        else:
            # Convert caption (string) to word ids.
            tokens = ['<start>', ]
            tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
            tokens.append('<end>')
            deleted_idx = []
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < 0.20:
                    prob /= 0.20
                    # 50% randomly change token to mask token
                    if prob < 0.5:
                        tokens[i] = self.vocab.word2idx['<mask>']
                    # 10% randomly change token to random token
                    elif prob < 0.6:
                        tokens[i] = random.randrange(len(self.vocab))
                    # 40% randomly remove the token
                    else:
                        tokens[i] = self.vocab(token)
                        deleted_idx.append(i)
                else:
                    tokens[i] = self.vocab(token)
            if len(deleted_idx) != 0:
                tokens = [tokens[i] for i in range(len(tokens)) if i not in deleted_idx]
            target = torch.Tensor(tokens)
            return target

    def process_image(self, image, img_enhance):
        enhance = img_enhance if self.data_split == 'train' else False
        if enhance:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            tmp = image[np.where(rand_list > 0.20)]
            while tmp.size(1) <= 1:
                rand_list = np.random.rand(num_features)
                tmp = image[np.where(rand_list > 0.20)]
            return tmp
        else:
            return image

    def process_image_2(self, image, img_enhance):
        enhance = img_enhance if self.data_split == 'train' else False
        if enhance:  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            tmp = image[np.where(rand_list > 0.20)]
            while tmp.size(1) <= 1:
                rand_list = np.random.rand(num_features)
                tmp = image[np.where(rand_list > 0.20)]
            image[np.where(rand_list < 0.20)] = 1e-10
            return image
        else:
            return image

    def __getitem__(self, index):
        # handle the image redundancy
        caption = self.captions[index]
        if self.data_split == 'train':
            if self.noise_type == 'DECL':
                img_id = self.noisy_inx[index // self.im_div]
            else:
                img_id = self.noisy_inx[index]  # NCR
        else:
            img_id = index // self.im_div
        image = torch.Tensor(self.images[img_id])
        # Convert caption (string) to word ids.
        a = True
        if self.module == 'VSEinfty':
            target = self.process_caption(caption, caption_enhance=a)
            image = self.process_image(image, img_enhance=a)
        else:
            target = self.process_caption(caption, caption_enhance=a)
            image = self.process_image_2(image, img_enhance=a)

        return image, target, index

    def __len__(self):
        return self.length

    def get_real_correspondence(self):
        return self.real_correspondence


def collate_fn(data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids = zip(*data)

    img_lengths = [len(image) for image in images]
    all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
    for i, image in enumerate(images):
        end = img_lengths[i]
        all_images[i, :end] = image[:end]
    img_lengths = torch.Tensor(img_lengths)
    # Merget captions
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    lengths = torch.Tensor(lengths)
    return all_images, img_lengths, targets, lengths, list(ids)


def get_precomp_loader(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2, logger=None):
    if shuffle:
        dset = PrecompDataset(data_path, data_split, vocab, opt, logger)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=False,
                                                  num_workers=num_workers,
                                                  collate_fn=collate_fn)
        return data_loader
    else:
        dset = PrecompDataset(data_path, data_split, vocab, opt, logger)

        real_correspondence = dset.get_real_correspondence()

        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=False,
                                                  num_workers=num_workers,
                                                  collate_fn=collate_fn)
        # return data_loader, real_correspondence

        return data_loader


def get_loaders(data_name, vocab, batch_size, workers, opt, logger):
    # get the data path
    dpath = os.path.join(opt.data_path, data_name)

    # get the train_loader
    train_loader = get_precomp_loader(dpath, 'train', vocab, opt,
                                      batch_size, True, workers, logger)
    # get the val_loader
    val_loader = get_precomp_loader(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers, logger)
    # get the test_loader
    test_loader = get_precomp_loader(dpath, 'test', vocab, opt,
                                     batch_size, False, workers, logger)
    return train_loader, val_loader, test_loader


def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    # get the data path
    data_path = "/data"
    dpath = os.path.join(data_path, data_name)

    # get the test_loader
    test_loader = get_precomp_loader(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
