import json
import os
import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='latin1')
    return dict


class cifar_dataset(Dataset):
    def __init__(self, dataset, r, noise_mode, root_dir, transform, mode,
                 clean=False, num_clean=1000, num_valid=5000):

        self.r = r  # noise ratio
        self.transform = transform
        self.mode = mode
        self.clean = clean
        self.num_clean = num_clean
        self.num_valid = num_valid

        # class transition for asymmetric noise
        self.transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}

        # test
        if self.mode == 'test':
            if dataset == 'cifar10':
                test_dic = unpickle('%s/test_batch' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['labels']
                self.num_classes = 10

            elif dataset == 'cifar100':
                test_dic = unpickle('%s/test' % root_dir)
                self.test_data = test_dic['data']
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic['fine_labels']
                self.num_classes = 100

        # train
        else:
            train_data = []
            train_label = []

            # load dataset
            if dataset == 'cifar10':
                for n in range(1, 6):
                    dpath = '%s/data_batch_%d' % (root_dir, n)
                    data_dic = unpickle(dpath)
                    train_data.append(data_dic['data'])
                    train_label = train_label + data_dic['labels']
                train_data = np.concatenate(train_data)
                img_num_list_meta = [int(self.num_clean / 10)] * 10
                img_num_list_val = [int(self.num_valid / 10)] * 10
                self.num_classes = 10

            elif dataset == 'cifar100':
                train_dic = unpickle('%s/train' % root_dir)
                train_data = train_dic['data']
                train_label = train_dic['fine_labels']
                train_coarse_label = train_dic['coarse_labels']
                img_num_list_meta = [int(self.num_clean / 100)] * 100
                img_num_list_val = [int(self.num_valid / 100)] * 100
                self.num_classes = 100

            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.idx_to_meta = []
            self.idx_to_train = []
            self.idx_to_valid = []

            data_list_val = {}
            for j in range(self.num_classes):
                data_list_val[j] = [i for i, label in enumerate(train_label) if label == j]

            for cls_idx, img_id_list in data_list_val.items():
                np.random.shuffle(img_id_list)
                num_meta = img_num_list_meta[int(cls_idx)]
                num_val = img_num_list_val[int(cls_idx)]
                self.idx_to_meta.extend(img_id_list[:num_meta])
                self.idx_to_train.extend(img_id_list[num_meta:-num_val])
                self.idx_to_valid.extend(img_id_list[-num_val:])

            if mode == "train_noisy":
                self.data_idx = self.idx_to_train
            elif mode == "train_clean":
                self.data_idx = self.idx_to_meta
            elif mode == "valid":
                self.data_idx = self.idx_to_valid

            self.train_data = train_data[self.data_idx]
            self.train_label = list(np.array(train_label)[self.data_idx])
            self.clean_label = (self.train_label).copy()

            self.label_dict = {}
            for j in range(self.num_classes):
                self.label_dict[j] = [i for i, label in enumerate(self.train_label) if label == j]

            if mode == "train_noisy":
                if dataset == 'cifar100' and noise_mode == 'asym':
                    self.train_coarse_label = list(np.array(train_coarse_label)[self.data_idx])
                    C_ = self.get_hierarchical(self.train_label, self.train_coarse_label)

                # make label corruption
                noise_label = []
                num_train = len(self.train_data)
                idx = list(range(num_train))
                random.shuffle(idx)
                num_noise = int(self.r * num_train)
                noise_idx = idx[:num_noise]
                for i in range(num_train):
                    if i in noise_idx:
                        if noise_mode == 'sym':
                            if dataset == 'cifar10':
                                noiselabel = random.randint(0, 9)
                            elif dataset == 'cifar100':
                                noiselabel = random.randint(0, 99)
                            noise_label.append(noiselabel)
                        elif noise_mode == 'asym':
                            if dataset == 'cifar10':
                                noiselabel = self.transition[self.train_label[i]]
                                noise_label.append(noiselabel)
                            elif dataset == 'cifar100':
                                noiselabel = np.random.choice(self.num_classes, p=C_[self.train_label[i]])
                                noise_label.append(noiselabel)
                    else:
                        noise_label.append(self.train_label[i])
                self.train_label = noise_label

    def get_hierarchical(self, train_label, train_coarse_label):
        assert self.num_classes == 100, 'You must use CIFAR-100 with the hierarchical corruption.'
        coarse_fine = []
        for i in range(20):
            coarse_fine.append(set())
        for i in range(len(train_label)):
            coarse_fine[train_coarse_label[i]].add(train_label[i])
        for i in range(20):
            coarse_fine[i] = list(coarse_fine[i])

        C = np.eye(self.num_classes) * (1 - self.r)
        C_ = np.zeros_like(C)
        for i in range(20):
            tmp = np.copy(coarse_fine[i])
            for j in range(len(tmp)):
                tmp2 = np.delete(np.copy(tmp), j)
                C[tmp[j], tmp2] += self.r * 1 / len(tmp2)
                C_[tmp[j], tmp2] += self.r * 1 / len(tmp2)

        for cls in range(len(C_)):
            C_[cls, :] /= np.sum(C_[cls, :])
        return C_

    def __getitem__(self, index):
        if self.mode == 'train_noisy':
            img, target, target_raw \
                = self.train_data[index], self.train_label[index], self.clean_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target, index
        elif self.mode == "train_clean" or self.mode == "valid":
            img, target = self.train_data[index], self.train_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target
        elif self.mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode != 'test':
            return len(self.train_data)
        else:
            return len(self.test_data)


class cifar_dataloader():
    def __init__(self, dataset, r, noise_mode, batch_size, num_workers, root_dir):
        self.dataset = dataset
        self.r = r
        self.noise_mode = noise_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.idx_to_meta = None

        if self.dataset == 'cifar10':
            self.num_classes = 10
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif self.dataset == 'cifar100':
            self.num_classes = 100
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
            ])

    def run(self, mode, args=None):
        if mode == 'train':
            use_valid = args.use_valid
            noisy_dataset = cifar_dataset(
                dataset=self.dataset,
                clean=False, num_clean=args.num_clean,
                noise_mode=self.noise_mode,
                r=self.r, root_dir=self.root_dir,
                transform=self.transform_train,
                mode="train_noisy",
            )
            clean_dataset = cifar_dataset(
                dataset=self.dataset,
                clean=True, num_clean=args.num_clean,
                noise_mode=self.noise_mode,
                r=self.r, root_dir=self.root_dir,
                transform=self.transform_train,
                mode="train_clean",
            )
            trainloader = DataLoader(
                dataset=noisy_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
            train_sampler = ImageSampler(
                data_source=clean_dataset,
                nPerImage=args.nPerImage,
                num_classes=args.num_classes)
            trainloader_c = torch.utils.data.DataLoader(
                clean_dataset,
                batch_size=args.num_classes * args.nPerImage,
                num_workers=self.num_workers,
                sampler=train_sampler,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
                drop_last=True,
            )
            if use_valid:
                valid_dataset = cifar_dataset(
                    dataset=self.dataset,
                    clean=True, num_clean=args.num_clean,
                    noise_mode=self.noise_mode,
                    r=self.r, root_dir=self.root_dir,
                    transform=self.transform_test,
                    mode="valid",
                )
                validloader = DataLoader(
                    dataset=valid_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers
                )
                return trainloader, trainloader_c, validloader
            else:
                return trainloader, trainloader_c

        elif mode == 'test':
            test_dataset = cifar_dataset(
                dataset=self.dataset,
                noise_mode=self.noise_mode,
                r=self.r,
                root_dir=self.root_dir,
                transform=self.transform_test,
                mode='test'
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return test_loader

        elif mode == 'clean':
            clean_dataset = cifar_dataset(
                dataset=self.dataset,
                clean=True, num_clean=args.num_clean,
                noise_mode=self.noise_mode,
                r=self.r, root_dir=self.root_dir,
                transform=self.transform_train,
                mode="train_clean",
            )
            clean_loader = DataLoader(
                dataset=clean_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            return clean_loader


class ImageSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerImage, num_classes):
        self.data_source = data_source
        self.label_dict = data_source.label_dict
        self.nPerImage = nPerImage
        self.batch_size = nPerImage * num_classes

    def __iter__(self):
        dictkeys = list(self.label_dict.keys());
        dictkeys.sort()

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for label in dictkeys:
            data = self.label_dict[label]
            data = random.sample(data, self.nPerImage)
            flattened_label.extend([label] * self.nPerImage)
            flattened_list.extend(data)

        return iter(flattened_list)

    def __len__(self):
        return len(self.data_source)
