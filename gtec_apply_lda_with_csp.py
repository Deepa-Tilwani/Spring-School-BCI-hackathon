import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score

from mne.channels import read_layout
from mne.decoding import CSP
import os
import mne
print(__doc__)

# subjects = [
#     'P1_pre_training',
#     'P1_pre_test',
#     'P1_post_training',
#     'P1_post_test',
#     'P2_pre_training',
#     'P2_pre_test',
#     'P2_post_training',
#     'P2_post_test',
#     'P3_pre_training',
#     'P3_pre_test',
#     'P3_post_training',
#     'P3_post_test'
# ]

subjects = [
    'P1_pre',
    'P1_post',
    'P2_pre',
    'P2_post',
    'P3_pre',
    'P3_post',
]
def read_epochs(subject_path):
    # #############################################################################
    # # Set parameters and read data

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs_list = []
    for (dirname, dirs, files) in os.walk(subject_path):
        for filename in files:
            if '-epo' in filename and '_mi_' in filename:
                epochs_list.append(mne.read_epochs(os.path.join(subject_path, filename), preload=True))
    for ep in epochs_list:
        ep.rename_channels(lambda x: x.strip(' '))
        ep.pick(picks='eeg')
    epochs = mne.concatenate_epochs(epochs_list)
    del epochs_list
    return epochs


def read_epochs_mi_mov(subject_path):
    # #############################################################################
    # # Set parameters and read data

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    train_epochs_list = []
    test_epochs_list = []
    for (dirname, dirs, files) in os.walk(subject_path):
        for filename in files:
            if '-epo' in filename:
                if 'test' in filename:
                    test_epochs_list.append(mne.read_epochs(os.path.join(subject_path, filename), preload=True))
                elif 'training' in filename:
                    train_epochs_list.append(mne.read_epochs(os.path.join(subject_path, filename), preload=True))
    for ep in train_epochs_list:
        ep.rename_channels(lambda x: x.strip(' '))
        ep.pick(picks='eeg')
    for ep in test_epochs_list:
        ep.rename_channels(lambda x: x.strip(' '))
        ep.pick(picks='eeg')
    if train_epochs_list:
        train_epochs = mne.concatenate_epochs(train_epochs_list)
    else:
        train_epochs = None
    if test_epochs_list:
        test_epochs = mne.concatenate_epochs(test_epochs_list)
    else:
        test_epochs = None
    del train_epochs_list, test_epochs_list
    return train_epochs, test_epochs

def read_epochs_mi_mov_train(subject_path):
    # #############################################################################
    # # Set parameters and read data

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    train_epochs_list = []
    for (dirname, dirs, files) in os.walk(subject_path):
        for filename in files:
            if '-epo' in filename:
                if 'training' in filename:
                    train_epochs_list.append(mne.read_epochs(os.path.join(subject_path, filename), preload=True))
    for ep in train_epochs_list:
        ep.rename_channels(lambda x: x.strip(' '))
        ep.pick(picks='eeg')

    if train_epochs_list:
        train_epochs = mne.concatenate_epochs(train_epochs_list)
    else:
        train_epochs = None

    del train_epochs_list
    return train_epochs

def read_epochs_mi_mov_test(subject_path):
    # #############################################################################
    # # Set parameters and read data

    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    test_epochs_list = []
    for (dirname, dirs, files) in os.walk(subject_path):
        for filename in files:
            if '-epo' in filename:
                if 'test' in filename:
                    test_epochs_list.append(mne.read_epochs(os.path.join(subject_path, filename), preload=True))

    for ep in test_epochs_list:
        ep.rename_channels(lambda x: x.strip(' '))
        ep.pick(picks='eeg')

    if test_epochs_list:
        test_epochs = mne.concatenate_epochs(test_epochs_list)
    else:
        test_epochs = None
    del test_epochs_list
    return test_epochs


def lda_with_csp(subject_name, epochs_train, epochs_test):
    # avoid classification of evoked responses by using epochs that start 1s after
    # cue onset.

    # #############################################################################
    # Apply LDA with CSP
    # Define a monte-carlo cross-validation generator (reduce variance):
    if not epochs_train:
        return None

    labels = epochs_train.events[:, -1]
    test_labels = epochs_test.events[:, -1]

    scores = []
    epochs_data = epochs_train.get_data()

    # epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
    epinfo = epochs_train.info
    # del epochs
    epochs_data_train = epochs_train.get_data()
    epochs_data_test = epochs_test.get_data()
    del epochs_train
    cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    # epochs_data_train = epochs_data_train.reshape(epochs_data_train.shape[0], -1)
    # cv_split = cv.split(epochs_data_train)

    # Assemble a classifier
    lda = LinearDiscriminantAnalysis()
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

    # Use scikit-learn Pipeline with cross_val_score function
    clf = Pipeline([('CSP', csp), ('LDA', lda)])
    scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)

    # Printing the results
    class_balance = np.mean(labels == labels[0])
    class_balance = max(class_balance, 1. - class_balance)
    scores_mean = np.mean(scores)

    clf.fit_transform(epochs_data_train, test_labels)
    clf_score = clf.score(epochs_data_test, test_labels)

    print('\n###########\n')
    print("Classification accuracy: %f / Chance level: %f" % (scores_mean,
                                                              class_balance))
    print(f"CLF Score: {clf_score}")
    print('\n###########\n')
    del epochs_data_train, cv, lda, scores, class_balance
    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(epochs_data, labels)

    layout = read_layout('EEG1005')
    export_title = subject_name + ':: ' + str(scores_mean) + ':: ' + str(clf_score)
    print(export_title)
    csp.plot_patterns(epinfo, layout=layout, ch_type='eeg',
                      units='Patterns (AU)', size=1.5, show_names=True, title=export_title)

    return scores_mean, clf_score


if __name__ == "__main__":
    # test
    # generic_path = '/home/miha/workspace/MNE_dataset/no_ca_folder'
    subject_scores = [['Subject ID', 'LDA with CSP mean score']]
    # pick_channels = list(set(ch_names) - set(exclude_channels))
    generic_path = '/home/miha/gtec_hackaton/stroke/data/by_subject'
    for subject_name in subjects:
        subject_path = os.path.join(generic_path, subject_name)
        print("#########")
        print(subject_path)
        print("#########")
        train_epochs, test_epochs = read_epochs_mi_mov(subject_path)
        # epochs.pick(picks=pick_channels)
        subject_scores_mean = lda_with_csp(subject_name, train_epochs, test_epochs)
        subject_scores.append([subject_name, subject_scores_mean])

    # for root, dir, files in os.walk(generic_path):
    #     for subject_path in files:
    #         # subject_path = os.path.join(generic_path, subject_name)
    #         subject_name = os.path.basename(subject_path).replace(".fif", "")
    #         print("#########")
    #         print(subject_path, subject_name)
    #         print("#########")
    #
    #         epochs = read_epochs_mi_mov(os.path.join(root, subject_path))
    #         # epochs.pick(picks=pick_channels)
    #         subject_scores_mean = lda_with_csp(epochs, subject_name)
    #         subject_scores.append([subject_name, subject_scores_mean])
