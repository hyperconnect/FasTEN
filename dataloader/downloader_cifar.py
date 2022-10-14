import os

from torchvision.datasets.utils import download_url, check_integrity

dataset = 'cifar10'


class CifarDownloader(object):
    def __init__(self, root='./', dataset='cifar10'):
        self.root = root
        if dataset == 'cifar10':
            self.base_folder = 'cifar-10-batches-py'
            self.url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            self.filename = "cifar-10-python.tar.gz"
            self.tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
            self.train_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]
            self.test_list = [
                ['test_batch', '40351d587109b95175f43aff81a1287e'],
            ]
        elif dataset == 'cifar100':
            self.base_folder = 'cifar-100-python'
            self.url = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            self.filename = "cifar-100-python.tar.gz"
            self.tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
            self.train_list = [
                ['train', '16019d7e3df5f24257cddd939b257f8d'],
            ]
            self.test_list = [
                ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
            ]

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)


if __name__ == '__main__':
    cifar_downloader = CifarDownloader('/path/to/dataset', dataset='cifar10')
    cifar_downloader.download()
