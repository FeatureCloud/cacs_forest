import pickle
import pandas as pd
import numpy as np
import threading
import time

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from FeatureCloud.app.engine.app import AppState, app_state, Role, LogLevel, State

SPLITS = 5

@app_state('initial', Role.BOTH)
class InitialState(AppState):
    """
    Initialize client.
    """

    def register(self):
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        self.log("[CLIENT] Initializing")
        if self.id is not None:  # Test if setup has happened already
            self.log("[CLIENT] Coordinator {self.is_coordinator}")
        
        return 'read input'
        
        
@app_state('read input', Role.BOTH)
class ReadInputState(AppState):
    """
    Read input data.
    """
    
    def register(self):
        self.register_transition('preprocessing', Role.BOTH)
        self.register_transition('read input', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            self.store('INPUT_DIR', "/mnt/input")
            self.store('OUTPUT_DIR', "/mnt/output")
        
            data = pd.read_csv(f"{self.load('INPUT_DIR')}/calc.csv", sep=';')
            print(f'Finished reading data')
            print(f'Samples: {len(data)}')
        
            self.store('data', data)
        
            return 'preprocessing'
        
        except Exception as e:
            self.log('error read input', LogLevel.ERROR)
            self.update(message='error read input', state=State.ERROR)
            print(e)
            return 'read input'
            
            
@app_state('preprocessing', Role.BOTH)
class PreprocessingState(AppState):
    """
    Preprocessing data.
    """
    
    def register(self):
        self.register_transition('training', Role.BOTH)
        self.register_transition('preprocessing', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            X, y = self.preprocessing(self.load('data'))
            print(f'Finished preprocessing')
            print(f'Samples: {len(X)}')
            self.store('iteration', 0)
            self.store('X', X)
            self.store('y', y)
            return 'training'
            
        except Exception as e:
            self.log('error preprocessing', LogLevel.ERROR)
            self.update(message='error preprocessing', state=State.ERROR)
            print(e)
            return 'preprocessing'
        
    def preprocessing(self, df):
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

        return X.astype(np.float64), y.astype(bool)
        
        
@app_state('training', Role.BOTH)
class TrainingState(AppState):
    """
    Train Random Forest Classifier and send them to the coordinator.
    """
    
    def register(self):
        self.register_transition('wait_1', Role.PARTICIPANT)
        self.register_transition('gather_1', Role.COORDINATOR)
        self.register_transition('training', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            rs = KFold(n_splits=SPLITS, shuffle=True, random_state=0)
            self.store('rs', rs)
            clf_list  = []
            for train_index, _ in rs.split(self.load('X')):
                print(f'Train clf...')
                clf = RandomForestClassifier()
                setattr(clf, 'client_id', self.id)
                clf.fit(self.load('X')[train_index], self.load('y')[train_index])
                clf_list.append(clf)
            self.store('clf', clf_list)
            print(f'Finished training')
            
            co = pickle.dumps({'client_id': self.id, 'clfs': self.load('clf')})
            self.send_data_to_coordinator(co)
            
            if self.is_coordinator:
                return 'gather_1'
            else:
                return 'wait_1'
        
        except Exception as e:
            self.log('error training', LogLevel.ERROR)
            self.update(message='error training', state=State.ERROR)
            print(e)
            return 'training'
    
    
@app_state('gather_1', Role.COORDINATOR)
class Gather1State(AppState):
    """
    The coordinator receives the trees from each client and broadcasts all trees to the clients.
    """
    
    def register(self):
        self.register_transition('evaluate', Role.COORDINATOR)
        self.register_transition('gather_1', Role.COORDINATOR)
        
    def run(self) -> str or None:
        try:
            data = self.gather_data()
            print(f'Received all trees')
            clfs = [pickle.loads(clfs) for clfs in data]
            self.store('clfs', clfs)
            self.broadcast_data(pickle.dumps(clfs), send_to_self=False)
            return 'evaluate'

        except Exception as e:
            self.log('error gather_1', LogLevel.ERROR)
            self.update(message='error gather_1', state=State.ERROR)
            print(e)
            return 'gather_1'


@app_state('wait_1', Role.PARTICIPANT)
class Wait1State(AppState):
    """
    The participant waits until it receives the random forests from the coordinator.
    """
    
    def register(self):
        self.register_transition('evaluate', Role.PARTICIPANT)
        self.register_transition('wait_1', Role.PARTICIPANT)
    
    def run(self) -> str or None:
        try:
            data = self.await_data()
            clfs = pickle.loads(data)
            self.store('clfs', clfs)
            return 'evaluate'

        except Exception as e:
            self.log('error wait_1', LogLevel.ERROR)
            self.update(message='error wait_1', state=State.ERROR)
            print(e)
            return 'wait_1'


@app_state('evaluate', Role.BOTH)
class EvaluateState(AppState):
    """
    Compose random forests using all combinations.
    """
    
    def register(self):
        self.register_transition('gather_2', Role.COORDINATOR)
        self.register_transition('terminal', Role.PARTICIPANT)
        self.register_transition('evaluate', Role.BOTH)
        
    def run(self) -> str or None:
        try:
            print(f"Have {len(self.load('clfs'))} random forests")
            print(f"RFs: {self.load('clfs')}")
        
            # {self.clfs} contains lists of random forests, one list per client
            # Each list, in turn, contains one random forest trained per split
            # We now compose random forests using all combinations as described in the paper

            own_clf = None
            other_clfs = []
            for clfs in self.load('clfs'):
                if clfs["client_id"] == self.id:
                    own_clf = clfs["clfs"]
                    continue
                other_clfs.append(clfs["clfs"])

            # {own_clf} is the list of own random forests
            # {other_clf} is a list of lists containing RFs from other clients (as described above)

            cm = np.array([[0, 0], [0, 0]])

            i = 0
            ie = 0
            total = SPLITS ** len(self.load('clfs'))
            for _, test_index in self.load('rs').split(self.load('X')):
                own_rf = own_clf[i]
                for idxs in self.generate_idx(len(other_clfs), SPLITS):
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
                    y_pred = new_rf.predict(self.load('X')[test_index])
                    cm = cm + confusion_matrix(self.load('y')[test_index], y_pred)
                i += 1

            print(f'Confusion matrix: {cm}')
            self.send_data_to_coordinator(pickle.dumps(cm))
            if self.is_coordinator:
                return 'gather_2'
            else:
                return 'terminal'
                
        except Exception as e:
            self.log('error evaluate', LogLevel.ERROR)
            self.update(message='error evaluate', state=State.ERROR)
            print(e)
            return 'evaluate'

    def generate_idx(self, dim, num):
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


@app_state('gather_2', Role.COORDINATOR)
class Gather2State(AppState):
    """
    Compute accuracy.
    """
    
    def register(self):
        self.register_transition('terminal', Role.COORDINATOR)
        self.register_transition('gather_2', Role.COORDINATOR)
    
    def run(self) -> str or None:
        try:
            data = self.gather_data()
            cm = np.array([[0, 0], [0, 0]])
            for c in data:
                cm = cm + pickle.loads(c)

            acc = (cm[0, 0] + cm[1, 1]) / cm.sum()

            print(f'Accuracy: {acc}')

            return 'terminal'
        
        except Exception as e:
            self.log('error gather_2', LogLevel.ERROR)
            self.update(message='error gather_2', state=State.ERROR)
            print(e)
            return 'gather_2'
