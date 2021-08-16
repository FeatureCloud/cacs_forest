import pickle
import pandas as pd
import numpy as np
import threading
import time

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

SPLITS = 5


class AppLogic:

    def __init__(self):
        # === Status of this app instance ===

        # Indicates whether there is data to share, if True make sure self.data_out is available
        self.status_available = False

        # Only relevant for coordinator, will stop execution when True
        self.status_finished = False

        # === Parameters set during setup ===
        self.id = None
        self.coordinator = None
        self.clients = None

        # === Data ===
        self.data_incoming = []
        self.data_outgoing = None

        # === Internals ===
        self.thread = None
        self.iteration = 0
        self.progress = 'not started yet'
        self.INPUT_DIR = "/mnt/input"
        self.OUTPUT_DIR = "/mnt/output"

        # === Custom ===
        self.rs = None
        self.data = None
        self.X = None
        self.y = None
        self.clf = []
        self.clfs = []

        self.client = None

    def handle_setup(self, client_id, coordinator, clients):
        # This method is called once upon startup and contains information about the execution context of this instance
        self.id = client_id
        self.coordinator = coordinator
        self.clients = clients
        print(f'Received setup: {self.id} {self.coordinator} {self.clients}', flush=True)

        self.thread = threading.Thread(target=self.app_flow)
        self.thread.start()

    def handle_incoming(self, data):
        # This method is called when new data arrives
        print("Process incoming data....", flush=True)
        self.data_incoming.append(data.read())

    def handle_outgoing(self):
        print("Process outgoing data...", flush=True)
        # This method is called when data is requested
        self.status_available = False
        return self.data_outgoing

    def app_flow(self):
        # This method contains a state machine for the client and coordinator instance

        # === States ===
        state_initializing = 1
        state_read = 2
        state_preprocessing = 3
        state_training = 4
        state_gather_1 = 5
        state_wait_1 = 6
        state_evaluate = 7
        state_gather_2 = 8

        state_finishing = 10

        # Initial state
        state = state_initializing
        self.progress = 'initializing...'

        while True:
            time.sleep(5)

            if state == state_initializing:
                print(f'State: state_initializing')
                self.progress = 'initializing...'
                state = state_read
                continue

            if state == state_read:
                print(f'State: state_read')
                self.progress = 'reading...'
                self.data = pd.read_csv(f'{self.INPUT_DIR}/calc.csv', sep=';')
                print(f'Finished reading data')
                print(f'Samples: {len(self.data)}')
                state = state_preprocessing
                continue

            if state == state_preprocessing:
                print(f'State: state_preprocessing')
                self.progress = 'preprocessing...'
                self.X, self.y = preprocessing(self.data)
                print(f'Finished preprocessing')
                print(f'Samples: {len(self.X)}')
                state = state_training
                continue

            if state == state_training:
                print(f'State: state_training')
                self.rs = KFold(n_splits=SPLITS, shuffle=True, random_state=0)
                for train_index, _ in self.rs.split(self.X):
                    print(f'Train clf...')
                    clf = RandomForestClassifier()
                    setattr(clf, 'client_id', self.id)
                    clf.fit(self.X[train_index], self.y[train_index])
                    self.clf.append(clf)
                print(f'Finished training')

                co = pickle.dumps({'client_id': self.id, 'clfs': self.clf})

                if self.coordinator:
                    self.data_incoming.append(co)
                    state = state_gather_1
                else:
                    self.data_outgoing = co
                    self.status_available = True
                    state = state_wait_1
                continue

            if state == state_gather_1:
                print(f'State: state_gather')
                if len(self.data_incoming) == len(self.clients):
                    print(f'Received all trees')
                    self.clfs = [pickle.loads(clfs) for clfs in self.data_incoming]
                    self.data_incoming = []
                    self.data_outgoing = pickle.dumps(self.clfs)
                    self.status_available = True
                    state = state_evaluate
                continue

            if state == state_wait_1:
                if len(self.data_incoming) == 1:
                    self.clfs = pickle.loads(self.data_incoming[0])
                    self.data_incoming = []
                    state = state_evaluate
                continue

            if state == state_evaluate:
                print(f'Have {len(self.clfs)} random forests')
                print(f'RFs: {self.clfs}')

                # {self.clfs} contains lists of random forests, one list per client
                # Each list, in turn, contains one random forest trained per split
                # We now compose random forests using all combinations as described in the paper

                own_clf = None
                other_clfs = []
                for clfs in self.clfs:
                    if clfs["client_id"] == self.id:
                        own_clf = clfs["clfs"]
                        continue
                    other_clfs.append(clfs["clfs"])

                # {own_clf} is the list of own random forests
                # {other_clf} is a list of lists containing RFs from other clients (as described above)

                cm = np.array([[0, 0], [0, 0]])

                i = 0
                ie = 0
                total = SPLITS ** len(self.clfs)
                for _, test_index in self.rs.split(self.X):
                    own_rf = own_clf[i]
                    for idxs in generate_idx(len(other_clfs), SPLITS):
                        new_rf = sklearn.base.clone(own_rf)
                        new_rf.n_estimators = own_rf.n_estimators
                        new_rf.estimators_ = own_rf.estimators_
                        new_rf.n_classes_ = own_rf.n_classes_
                        new_rf.n_outputs_ = own_rf.n_outputs_
                        new_rf.classes_ = own_rf.classes_
                        for i, idx in enumerate(idxs):
                            new_rf.n_estimators += other_clfs[i][idx].n_estimators
                            new_rf.estimators_ += other_clfs[i][idx].estimators_
                        ie += 1
                        print(f'Evaluate {ie}/{total}')
                        y_pred = new_rf.predict(self.X[test_index])
                        cm = cm + confusion_matrix(self.y[test_index], y_pred)
                    i += 1

                print(f'Confusion matrix: {cm}')

                if self.coordinator:
                    self.data_incoming.append(pickle.dumps(cm))
                    state = state_gather_2
                else:
                    self.data_outgoing = pickle.dumps(cm)
                    self.status_available = True
                    state = state_finishing

                continue

            if state == state_gather_2:
                if len(self.data_incoming) == len(self.clients):
                    cm = np.array([[0, 0], [0, 0]])
                    for c in self.data_incoming:
                        cm = cm + pickle.loads(c)

                    acc = (cm[0, 0] + cm[1, 1]) / cm.sum()

                    print(f'Accuracy: {acc}')

                    state = state_finishing

                continue

            if state == state_finishing:
                print(f'State: state_finishing')
                self.progress = 'finishing...'
                time.sleep(10)
                self.status_finished = True
                break


logic = AppLogic()


def preprocessing(df):
    # Remove missing values
    df = df[df['kalk'].notnull()]
    df = df[df['height'].notna()]
    df = df[df['weight'].notna()]
    df = df[df['waist'].notna()]
    df = df[df['chol'].notna()]
    df = df[df['tri'].notna()]
    df = df[df['hdl'].notna()]
    df = df[df['ldl'].notna()]
    df = df[df['hba'].notna()]
    df['bmi'] = df['weight'] * 10000 / df['height'] / df['height']

    df['age'] = 2015 - df['birth_year']

    df.sex = pd.Categorical(df.sex)
    df.sex = df.sex.cat.codes

    df.loc[df['kalk'] < 5, 'kalk_cat'] = False
    df.loc[df['kalk'] >= 5, 'kalk_cat'] = True

    a_count = len(df[df['kalk_cat'] == False].index)
    b_count = len(df[df['kalk_cat'] == True].index)
    total_count = a_count + b_count

    print()
    print(f'{total_count} samples in total:')
    print(f'Normal:   {a_count:4d} {a_count * 100 / total_count:2.2f}%')
    print(f'Elevated: {b_count:4d} {b_count * 100 / total_count:2.2f}%')

    X = df[['age', 'sex', 'height', 'weight', 'waist', 'chol', 'tri', 'hdl', 'ldl', 'hba', 'bmi']].values
    y = df['kalk_cat'].values

    return X.astype(np.float64), y.astype(np.bool)


def generate_idx(dim, num):
    vec = [0] * dim
    while True:
        yield vec
        for idx in range(dim):
            vec[idx] += 1
            if vec[idx] == num:
                if idx == dim - 1:
                    return
                vec[idx] = 0
            else:
                break
