__docformat__ = 'restructedtext en'

import warnings
from sklearn.model_selection import StratifiedKFold
import numpy as np

__author__ = "Miquel Perello Nieto"
__credits__ = ["Miquel Perello Nieto"]

__license__ = "GPL"
__version__ = "1.0"
__maintainer__ = "Miquel Perello Nieto"
__email__ = "miquel@perellonieto.com"
__status__ = "Development"

from os.path import isfile
import requests

dataset_names_binary = ['credit-approval', 'diabetes',
                        'german', 'heart-statlog', 'hepatitis',
                        'horse', 'ionosphere', 'lung-cancer',
                        'mushroom', 'scene-classification',
                        'sonar', 'spambase', 'tic-tac',
                        'wdbc', 'wpbc']

datasets_li2014 = ['abalone', 'balance-scale', 'credit-approval',
                   'dermatology', 'german', 'heart-statlog', 'hepatitis',
                   'horse', 'ionosphere', 'lung-cancer', 'libras-movement',
                   'mushroom', 'diabetes', 'landsat-satellite', 'segment',
                   'spambase', 'wdbc', 'wpbc']

datasets_hempstalk2008 = ['diabetes',
                          'heart-statlog', 'ionosphere', 'iris', 'letter',
                          'mfeat-karhunen', 'mfeat-morphological', 'mfeat-zernike',
                          'optdigits', 'pendigits', 'sonar', 'vehicle', 'waveform-5000']

datasets_others = ['diabetes', 'heart-statlog',
                   'ionosphere', 'iris', 'letter', 'mfeat-karhunen',
                   'mfeat-morphological', 'mfeat-zernike', 'optdigits',
                   'pendigits', 'sonar', 'vehicle', 'waveform-5000',
                   'scene-classification', 'tic-tac', 'autos', 'car', 'cleveland',
                   'dermatology', 'flare', 'page-blocks', 'segment', 'shuttle',
                   'vowel', 'abalone', 'balance-scale', 'credit-approval',
                   'german', 'hepatitis', 'lung-cancer', 'ecoli', 'glass', 'yeast', 'zoo']


class Dataset(object):
    def __init__(self, name, data, target):
        self.name = name
        self._data = self.standardize_data(data)
        self._target, self._classes, self._names, self._counts = self.standardize_targets(target)

    def standardize_data(self, data):
        new_data = data.astype(float)
        data_mean = new_data.mean(axis=0)
        data_std = new_data.std(axis=0)
        data_std[data_std == 0] = 1
        return (new_data - data_mean) / data_std

    def standardize_targets(self, target):
        target = np.squeeze(target)
        names, counts = np.unique(target, return_counts=True)
        new_target = np.empty_like(target, dtype=int)
        for i, name in enumerate(names):
            new_target[target == name] = i
        classes = range(len(names))
        return new_target, classes, names, counts

    def separate_sets(self, x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    def reduce_number_instances(self, proportion=0.1):
        skf = StratifiedKFold(self._target, n_folds=1.0 / proportion)
        test_folds = skf.test_folds
        _, _, self._data, self._target = self.separate_sets(
            self._data, self._target, 0, test_folds)

    @property
    def target(self):
        return self._target

    # @target.setter
    # def target(self, new_value):
    #    self._target = new_value

    @property
    def data(self):
        return self._data

    @property
    def names(self):
        return self._names

    @property
    def classes(self):
        return self._classes

    @property
    def counts(self):
        return self._counts

    def print_summary(self):
        print(self)

    @property
    def n_classes(self):
        return len(self._classes)

    def __str__(self):
        return ("Name = {}\n"
                "Data shape = {}\n"
                "Target shape = {}\n"
                "Target classes = {}\n"
                "Target labels = {}").format(self.name, self.data.shape,
                                             self.target.shape, self.classes,
                                             self.names)


class Data(object):
    uci_nan = -2147483648
    mldata_names = {'diabetes': 'diabetes',
                    'ecoli': 'uci-20070111 ecoli',
                    'glass': 'glass',
                    'heart-statlog': 'datasets-UCI heart-statlog',
                    'ionosphere': 'ionosphere',
                    'iris': 'iris',
                    'letter': 'letter',
                    'mfeat-karhunen': 'uci-20070111 mfeat-karhunen',
                    'mfeat-morphological': 'uci-20070111 mfeat-morphological',
                    'mfeat-zernike': 'uci-20070111 mfeat-zernike',
                    'optdigits': 'uci-20070111 optdigits',
                    'pendigits': 'uci-20070111 pendigits',
                    'sonar': 'sonar',
                    'vehicle': 'vehicle',
                    'waveform-5000': 'datasets-UCI waveform-5000',
                    'scene-classification': 'scene-classification',
                    'tic-tac': 'uci-20070111 tic-tac-toe',
                    'MNIST': 'MNIST (original)',
                    'autos': 'uci-20070111 autos',
                    'car': 'uci-20070111 car',
                    'cleveland': 'uci-20070111 cleveland',
                    'dermatology': 'uci-20070111 dermatology',
                    'flare': 'uci-20070111 solar-flare_2',
                    'page-blocks': 'uci-20070111 page-blocks',
                    'segment': 'datasets-UCI segment',
                    'shuttle': 'shuttle',
                    'vowel': 'uci-20070111 vowel',
                    'zoo': 'uci-20070111 zoo',
                    'abalone': 'uci-20070111 abalone',
                    'balance-scale': 'uci-20070111 balance-scale',
                    'credit-approval': 'uci-20070111 credit-a',
                    'german': 'German IDA',
                    'hepatitis': 'uci-20070111 hepatitis',
                    'lung-cancer': 'Lung Cancer (Michigan)'
                    }

    mldata_not_working = {
        # 'spam':'uci-20070111 spambase', # not working
        'mushroom': 'uci-20070111 mushroom',
        # To be added:
        'breast-cancer-w': 'uci-20070111 wisconsin',
        # Need preprocessing :
        'auslan': '',
        # Needs to be generated
        'led7digit': '',
        'yeast': '',
        # Needs permission from ml-repository@ics.uci.edu
        'lymphography': '',
        # HTTP Error 500 in mldata.org
        'satimage': 'satimage',
        'nursery': 'uci-20070111 nursery',
        'hypothyroid': 'uci-20070111 hypothyroid'
    }

    def __init__(self, data_home='./datasets/', dataset_names=None, load_all=False):
        self.data_home = data_home
        self.datasets = {}

        if load_all:
            dataset_names = Data.mldata_names.keys()
            self.load_datasets_by_name(dataset_names)
        elif dataset_names is not None:
            self.load_datasets_by_name(dataset_names)

    def load_datasets_by_name(self, names):
        for name in names:
            dataset = self.get_dataset_by_name(name)
            if dataset is not None:
                self.datasets[name] = self.get_dataset_by_name(name)
            else:
                warnings.simplefilter('always', UserWarning)
                warnings.warn(("Dataset '{}' not currently available.".format(name)),
                              UserWarning)

    def download_file_content(self, url):

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise an exception for non-2xx status codes
            return response.content
        except requests.exceptions.RequestException as e:
            raise  # Re-raise the exception for handling by the caller

    def save_file_content(self, filename, content):
        with open(filename, 'w') as f:
            f.write(content)

    def check_file_and_download(self, file_path, url):
        if not isfile(file_path):
            content = self.download_file_content(url)
            self.save_file_content(file_path, content)

    def get_dataset_by_name(self, name):
        if name in Data.mldata_names.keys():
            return self.get_mldata_dataset(name)
        elif name == 'spambase':
            file_path = self.data_home + 'spambase.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',')
            target = data[:, -1]
            data = data[:, 0:-1]
        elif name == 'horse':
            file_path = self.data_home + 'horse-colic.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path)
            target = data[:, 23]
            data = np.delete(data, 23, axis=1)
            data = self.substitute_missing_values(data, column_mean=True)
        elif name == 'libras-movement':
            file_path = self.data_home + 'movement_libras.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/libras/movement_libras.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',')
            target = data[:, -1]
            data = np.delete(data, -1, axis=1)
        elif name == 'mushroom':
            file_path = self.data_home + 'agaricus-lepiota.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',', dtype=np.object_)
            target = data[:, 0]
            data = np.delete(data, 0, axis=1)
            for i in range(data.shape[1]):
                data[:, i] = self.nominal_to_float(data[:, i])
            data = data.astype(float)
            data = self.substitute_missing_values(data, column_mean=True)
        elif name == 'landsat-satellite':
            file_path = self.data_home + 'sat.trn'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn"
            self.check_file_and_download(file_path, url)

            file_path = self.data_home + 'sat.tst'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst"
            self.check_file_and_download(file_path, url)

            data_train = np.genfromtxt(self.data_home + 'sat.trn')
            data_test = np.genfromtxt(self.data_home + 'sat.tst')

            target = np.hstack((data_train[:, -1], data_test[:, -1]))
            data = np.vstack((np.delete(data_train, -1, axis=1),
                              np.delete(data_test, -1, axis=1)))
        elif name == 'yeast':
            file_path = self.data_home + 'yeast.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
            self.check_file_and_download(file_path, url)

            target = np.genfromtxt(file_path, usecols=9, dtype=str)
            data = np.genfromtxt(self.data_home + 'yeast.data')[:, 1:-1]
        elif name == 'wdbc':
            file_path = self.data_home + 'wdbc.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',')
            data = np.delete(data, (0, 1), axis=1)
            target = np.genfromtxt(file_path, delimiter=',', usecols=1,
                                   dtype=str)
        elif name == 'wpbc':
            file_path = self.data_home + 'wpbc.data'
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data"
            self.check_file_and_download(file_path, url)

            data = np.genfromtxt(file_path, delimiter=',')
            data = np.delete(data, (0, 1), axis=1)
            target = np.genfromtxt(file_path, delimiter=',', usecols=1,
                                   dtype=str)
            data, target = self.remove_rows_with_missing_values(data, target)
        else:
            return None
        return Dataset(name, data, target)


    def mldata_to_numeric_matrix(self, mldata, n_samples, exclude=[]):
        """converts an mldata object into a matrix

        for each value in the mldata dictionary it is reshaped to contain the
        first dimension as a number of samples and the second as number of
        features. If the value contains numerical data it is not preprocessed.
        If the value contains any other type np.object_ it is transformed to
        numerical and all the missing values marked with '?' or 'nan' are
        substituted by np.nan.
        Args:
            mldata (dictionary with some numpy.array): feature strings.

        Returns:
            (array-like, shape = [n_samples, n_features]): floats.
        """
        first_column = True
        for key, submatrix in mldata.iteritems():
            if key not in exclude and type(submatrix) == np.ndarray:
                new_submatrix = np.copy(submatrix)

                if new_submatrix.shape[0] != n_samples:
                    new_submatrix = new_submatrix.T

                if new_submatrix.dtype.type == np.object_:
                    new_submatrix = self.nominal_to_float(new_submatrix)

                if first_column:
                    matrix = new_submatrix.reshape(n_samples, -1)
                    first_column = False
                else:
                    matrix = np.hstack((matrix,
                                        new_submatrix.reshape(n_samples, -1)))
        return matrix

    def nominal_to_float(self, x, missing_values=['nan', '?']):
        """converts an array of nominal features into floats

        Missing values are marked with the string 'nan' and are converted to
        numpy.nan

        Args:
            x (array-like, shape = [n_samples, 1]): feature strings.

        Returns:
            (array-like, shape = [n_samples, 1]): floats.
        """
        new_x = np.empty_like(x, dtype=float)
        x = np.squeeze(x)
        names = np.unique(x)
        substract = 0
        for i, name in enumerate(names):
            if name in missing_values:
                new_x[x == name] = np.nan
                substract += 1
            else:
                new_x[x == name] = i - substract
        return new_x

    def number_of_missing_values(self, data):
        return np.logical_or(np.isnan(data), data == self.uci_nan).sum()

    def row_indices_with_missing_values(self, data):
        return np.logical_or(np.isnan(data),
                             data == self.uci_nan).any(axis=1)

    def remove_rows_with_missing_values(self, data, target):
        missing = self.row_indices_with_missing_values(data)
        data = data[~missing]
        target = target[~missing]
        return data, target

    def remove_columns_with_missing_values(self, data, n_columns=1):
        for i in range(n_columns):
            index = np.isnan(data).sum(axis=0).argmax()
            data = np.delete(data, index, axis=1)
        return data

    def substitute_missing_values(self, data, fix_value=0, column_mean=False):
        for i in range(data.shape[1]):
            index = np.where(np.isnan(data[:, i]))
            if column_mean:
                mean = np.nanmean(data[:, i])
                data[index, i] = mean
            else:
                data[index, i] = fix_value
        return data

    def sumarize_datasets(self, name=None):
        if name is not None:
            dataset = self.datasets[name]
            dataset.print_summary()
        else:
            for name, dataset in self.datasets.iteritems():
                dataset.print_summary()


def test_datasets(dataset_names):
    from sklearn.svm import SVC
    data = Data(dataset_names=dataset_names)

    def separate_sets(x, y, test_fold_id, test_folds):
        x_test = x[test_folds == test_fold_id, :]
        y_test = y[test_folds == test_fold_id]

        x_train = x[test_folds != test_fold_id, :]
        y_train = y[test_folds != test_fold_id]
        return [x_train, y_train, x_test, y_test]

    n_folds = 2
    accuracies = {}
    for name, dataset in data.datasets.iteritems():
        dataset.print_summary()
        skf = StratifiedKFold(dataset.target, n_folds=n_folds, shuffle=True)
        test_folds = skf.test_folds
        accuracies[name] = np.zeros(n_folds)
        for test_fold in np.arange(n_folds):
            x_train, y_train, x_test, y_test = separate_sets(
                dataset.data, dataset.target, test_fold, test_folds)

            svc = SVC(C=1.0, kernel='rbf', degree=1, tol=0.01)
            svc.fit(x_train, y_train)
            prediction = svc.predict(x_test)
            accuracies[name][test_fold] = 100 * np.mean((prediction == y_test))
            print("Acc = {0:.2f}%".format(accuracies[name][test_fold]))
    return accuracies


def test():
    datasets_li2014 = ['abalone', 'balance-scale', 'credit-approval',
                       'dermatology', 'ecoli', 'german', 'heart-statlog', 'hepatitis',
                       'horse', 'ionosphere', 'lung-cancer', 'libras-movement',
                       'mushroom', 'diabetes', 'landsat-satellite', 'segment',
                       'spambase', 'wdbc', 'wpbc', 'yeast']

    datasets_hempstalk2008 = ['diabetes', 'ecoli', 'glass',
                              'heart-statlog', 'ionosphere', 'iris', 'letter',
                              'mfeat-karhunen', 'mfeat-morphological', 'mfeat-zernike',
                              'optdigits', 'pendigits', 'sonar', 'vehicle', 'waveform-5000']

    datasets_others = ['diabetes', 'ecoli', 'glass', 'heart-statlog',
                       'ionosphere', 'iris', 'letter', 'mfeat-karhunen',
                       'mfeat-morphological', 'mfeat-zernike', 'optdigits',
                       'pendigits', 'sonar', 'vehicle', 'waveform-5000',
                       'scene-classification', 'tic-tac', 'autos', 'car', 'cleveland',
                       'dermatology', 'flare', 'page-blocks', 'segment', 'shuttle',
                       'vowel', 'zoo', 'abalone', 'balance-scale', 'credit-approval',
                       'german', 'hepatitis', 'lung-cancer']

    # Datasets that we can add but need to be reduced
    datasets_to_add = ['MNIST']

    dataset_names = list(set(datasets_li2014 + datasets_hempstalk2008 +
                             datasets_others))

    accuracies = test_datasets(dataset_names)
    for i, name in enumerate(dataset_names):
        if name in accuracies.keys():
            print("{}. {} Acc = {:.2f}% +- {:.2f}".format(
                i + 1, name, accuracies[name].mean(), accuracies[name].std()))
        else:
            print("{}. {}  Not Available yet".format(i + 1, name))


class MLData(Data):
    def __init__(self, data_home='./datasets/', load_all=False):
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(('This Class is going to be deprecated in a future '
                       'version, please use cwc.data_wrappers.Data instead.'),
                      DeprecationWarning)
        self.data_home = data_home
        self.datasets = {}

        if load_all:
            for key in MLData.mldata_names.keys():
                self.datasets[key] = self.get_dataset(key)


if __name__ == '__main__':
    test()
