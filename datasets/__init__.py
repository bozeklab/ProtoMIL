import sys

from torch.utils.data import DataLoader

from datasets.breast_dataset import BreastCancerBagsCross
from datasets.camelyon_dataset import CamelyonPreprocessedBagsCross
from datasets.colon_dataset import ColonCancerBagsCross
from datasets.mnist_dataset import MnistBags
from datasets.messidor_dataset import DiabeticRetinopathyDataset
from datasets.rcc_dataset import RCCPreprocessedBagsCross
from datasets.mito_dataset import MitoPreprocessedBagsCross

PATH_TO_MESSIDOR_PATCH = '/shared/sets/datasets/vision/messidor/retina_patches/patches.csv'
PATH_TO_MESSIDOR_LABELS = '/shared/sets/datasets/vision/messidor/messidor_scaled_700x700/trainLabels.csv'


def get_train(name, seed=3, workers=0, config=None):
    if name == 'colon_cancer':
        ds = ColonCancerBagsCross(path="data/ColonCancer", train=True, shuffle_bag=True, data_augmentation=True,
                                  fold_id=config.fold_id, folds=config.folds, random_state=seed)
        ds_push = ColonCancerBagsCross(path="data/ColonCancer", train=True, push=True, shuffle_bag=True,
                                       fold_id=config.fold_id, folds=config.folds, random_state=seed)
    elif name == 'breast_cancer':
        ds = BreastCancerBagsCross(path="data/Bisque", train=True, shuffle_bag=True, data_augmentation=True,
                                   fold_id=config.fold_id, folds=config.folds, random_state=seed)
        ds_push = BreastCancerBagsCross(path="data/Bisque", train=True, push=True, shuffle_bag=True,
                                        fold_id=config.fold_id, folds=config.folds, random_state=seed)
    elif name == 'mito':
        ds = MitoPreprocessedBagsCross(path="data/mito_test_patches", train=True, shuffle_bag=True, data_augmentation=True,
                                   fold_id=config.fold_id, folds=config.folds, random_state=seed)
        ds_push = MitoPreprocessedBagsCross(path="data/mito_test_patches", train=True, push=True, shuffle_bag=True,
                                        fold_id=config.fold_id, folds=config.folds, random_state=seed)
    elif name == 'mnist':
        ds = MnistBags(train=True, random_state=seed, **config.dataset_settings)
        ds_push = MnistBags(train=True, push=True, random_state=seed, **config.dataset_settings)
    elif name == 'camelyon':
        ds = CamelyonPreprocessedBagsCross(path="data/CAMELYON_patches", train=True, shuffle_bag=True,
                                           data_augmentation=True,
                                           random_state=seed)
        ds_push = CamelyonPreprocessedBagsCross(path="data/CAMELYON_patches", train=True, push=True, shuffle_bag=True,
                                                random_state=seed)
    elif name == 'messidor':
        ds = DiabeticRetinopathyDataset(PATH_TO_MESSIDOR_PATCH, PATH_TO_MESSIDOR_LABELS, train=True, shuffle_bag=True,
                                        data_augmentation=True, fold_id=config.fold_id, folds=config.folds,
                                        random_state=seed)
        ds_push = DiabeticRetinopathyDataset(PATH_TO_MESSIDOR_PATCH, PATH_TO_MESSIDOR_LABELS, train=True, push=True,
                                             shuffle_bag=True, fold_id=config.fold_id, folds=config.folds,
                                             random_state=seed)
    elif name == 'rcc':
        ds = RCCPreprocessedBagsCross(path="data/RCC_patches", train=True, shuffle_bag=True,
                                      data_augmentation=True, random_state=seed, fold_id=config.fold_id)
        ds_push = RCCPreprocessedBagsCross(path="data/RCC_patches", train=True, push=True, shuffle_bag=True,
                                           random_state=seed, fold_id=config.fold_id)
    else:
        raise NotImplementedError()

    print('training set size: {}'.format(len(ds)))

    train_loader = DataLoader(
        ds, batch_size=None, shuffle=True,
        num_workers=workers,
        pin_memory=False)
    train_push_loader = DataLoader(
        ds_push, batch_size=None, shuffle=False,
        num_workers=workers,
        pin_memory=False)

    return train_loader, train_push_loader


def get_valid(name, seed=3, workers=0, config=None):
    if name == 'colon_cancer':
        ds_valid = ColonCancerBagsCross(path="data/ColonCancer", train=False, all_labels=True, fold_id=config.fold_id,
                                        folds=config.folds, random_state=seed)
        ds_push_valid = ColonCancerBagsCross(path="data/ColonCancer", train=False, all_labels=True,
                                             fold_id=config.fold_id,
                                             folds=config.folds, random_state=seed, push=True)

    elif name == 'breast_cancer':
        ds_valid = BreastCancerBagsCross(path="data/Bisque", train=False, all_labels=True, fold_id=config.fold_id,
                                         folds=config.folds, random_state=seed)
        ds_push_valid = BreastCancerBagsCross(path="data/Bisque", train=False, all_labels=True, fold_id=config.fold_id,
                                              folds=config.folds, random_state=seed, push=True, )

    elif name == 'mito':
        ds_valid = MitoPreprocessedBagsCross(path="data/mito_test_patches", train=False, all_labels=True, fold_id=config.fold_id,
                                         folds=config.folds, random_state=seed)
        ds_push_valid = MitoPreprocessedBagsCross(path="data/mito_test_patches", train=False, all_labels=True, fold_id=config.fold_id,
                                              folds=config.folds, random_state=seed, push=True, )

    elif name == 'mnist':
        ds_valid = MnistBags(train=False, random_state=seed, **config.dataset_settings, all_labels=True)
        ds_push_valid = MnistBags(train=False, random_state=seed, **config.dataset_settings, all_labels=True, push=True)

    elif name == 'camelyon':
        ds_valid = CamelyonPreprocessedBagsCross(path="data/CAMELYON_patches", train=False, all_labels=True,
                                                 fold_id=config.fold_id,
                                                 folds=config.folds, random_state=seed)
        ds_push_valid = CamelyonPreprocessedBagsCross(path="data/CAMELYON_patches", train=False, all_labels=True,
                                                      random_state=seed, push=True, )
    elif name == 'messidor':
        ds_valid = DiabeticRetinopathyDataset(PATH_TO_MESSIDOR_PATCH, PATH_TO_MESSIDOR_LABELS, train=False,
                                              fold_id=config.fold_id,
                                              folds=config.folds, random_state=seed)
        ds_push_valid = DiabeticRetinopathyDataset(PATH_TO_MESSIDOR_PATCH, PATH_TO_MESSIDOR_LABELS, train=False,
                                                   fold_id=config.fold_id,
                                                   folds=config.folds, random_state=seed, push=True)
    elif name == 'rcc':
        ds_valid = RCCPreprocessedBagsCross(path="data/RCC_patches", train=False, shuffle_bag=True,
                                      random_state=seed, fold_id=config.fold_id)
        ds_push_valid = RCCPreprocessedBagsCross(path="data/RCC_patches", train=False, push=True, shuffle_bag=True,
                                           random_state=seed, fold_id=config.fold_id)
    else:
        raise NotImplementedError()

    print('valid set size: {}'.format(len(ds_valid)))

    valid_loader = DataLoader(
        ds_valid, batch_size=None, shuffle=False,
        num_workers=workers,
        pin_memory=False)
    valid_push_loader = DataLoader(
        ds_push_valid, batch_size=None, shuffle=False,
        num_workers=workers,
        pin_memory=False)

    return valid_loader, valid_push_loader


def get_test(name, seed=3, workers=0, config=None):
    if name == 'colon_cancer':
        ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, test=True, all_labels=True,
                                       fold_id=config.fold_id, folds=config.folds, random_state=seed)
        ds_push_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, test=True, push=True, all_labels=True,
                                            fold_id=config.fold_id, folds=config.folds, random_state=seed)
    elif name == 'breast_cancer':
        ds_test = BreastCancerBagsCross(path="data/Bisque", train=False, test=True, all_labels=True,
                                        fold_id=config.fold_id, folds=config.folds, random_state=seed)
        ds_push_test = BreastCancerBagsCross(path="data/Bisque", train=False, test=True, push=True, all_labels=True,
                                             fold_id=config.fold_id, folds=config.folds, random_state=seed)
    elif name == 'mito':
        ds_test = MitoPreprocessedBagsCross(path="data/mito_test_patches", train=False, test=True, all_labels=True,
                                        fold_id=config.fold_id, folds=config.folds, random_state=seed)
        ds_push_test = MitoPreprocessedBagsCross(path="data/mito_test_patches", train=False, test=True, push=True, all_labels=True,
                                             fold_id=config.fold_id, folds=config.folds, random_state=seed)
    elif name == 'mnist':
        ds_test = MnistBags(train=False, test=True, random_state=seed, **config.dataset_settings, all_labels=True)
        ds_push_test = MnistBags(train=False, test=True, push=True, random_state=seed, **config.dataset_settings,
                                 all_labels=True)
    elif name == 'camelyon':
        ds_test = CamelyonPreprocessedBagsCross(path="data/CAMELYON_patches", train=False, test=True, all_labels=True,
                                                random_state=seed)
        ds_push_test = CamelyonPreprocessedBagsCross(path="data/CAMELYON_patches", train=False, test=True,
                                                     all_labels=True, push=True)
    elif name == 'rcc':
        ds_test = RCCPreprocessedBagsCross(path="data/RCC_patches", train=False, test=True, shuffle_bag=True,
                                      random_state=seed, fold_id=config.fold_id)
        ds_push_test = RCCPreprocessedBagsCross(path="data/RCC_patches", train=False, test=True, push=True, shuffle_bag=True,
                                           random_state=seed, fold_id=config.fold_id)
    elif name == 'messidor':
        ds_test = DiabeticRetinopathyDataset(PATH_TO_MESSIDOR_PATCH, PATH_TO_MESSIDOR_LABELS, train=False, test=True,
                                             fold_id=config.fold_id, folds=config.folds, random_state=seed)
        ds_push_test = DiabeticRetinopathyDataset(PATH_TO_MESSIDOR_PATCH, PATH_TO_MESSIDOR_LABELS, train=False,
                                                  test=True,
                                                  fold_id=config.fold_id, folds=config.folds, random_state=seed,
                                                  push=True)

    else:
        raise NotImplementedError()

    print('test set size: {}'.format(len(ds_test)))

    test_loader = DataLoader(
        ds_test, batch_size=None, shuffle=False,
        num_workers=workers,
        pin_memory=False)
    test_push_loader = DataLoader(
        ds_push_test, batch_size=None, shuffle=False,
        num_workers=workers,
        pin_memory=False)

    return test_loader, test_push_loader


def get_datasets(name, seed=3, workers=0, config=None):
    return (*get_train(name, seed, workers, config),
            *get_valid(name, seed, workers, config),
            *get_test(name, seed, workers, config))
