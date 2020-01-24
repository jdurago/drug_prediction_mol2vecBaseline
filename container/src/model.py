# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import ast
import argparse
import logging
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, confusion_matrix, precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib

import numpy as np

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

def load_model(model_path: str) -> word2vec.Word2Vec:
    return word2vec.Word2Vec.load(model_path)

def download_data(dev_mode: str, model: word2vec.Word2Vec) -> (np.ndarray, np.ndarray):
    """
    Returns tuple X, y which are numpy arrays
    """
    assert dev_mode.lower() == 'false' or dev_mode.lower() == 'true'
    
    if dev_mode.lower() == 'false':
        print('Using Actual Data...')
        data_path = os.path.join(args.data_dir, 'HIV.csv')
        df = pd.read_csv(data_path)
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(Chem.MolFromSmiles(x['smiles']), 1)), axis=1)
        df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], model, unseen='UNK')]
        
        # convert dataframe into numpy array for training
        X = np.array([x.vec for x in df['mol2vec']])
        y = np.array(df['HIV_active'].astype(int))
    else:
        # use example data set
        data_path = os.path.join(args.data_dir, 'ames.sdf')
        df = PandasTools.LoadSDF(data_path)
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
        df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], model, unseen='UNK')]
        
        # convert dataframe into numpy array for training
        X = np.array([x.vec for x in df['mol2vec']])
        y = np.array(df['class'].astype(int))
        
    return X,y

def split_data(X:np.ndarray, y:np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Returns np.ndarrays X_train, X_test, y_train, y_test
    """
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    return X_train, X_val, y_train, y_val

def generate_confusion_matrix_plot(clf: RandomForestClassifier, X_val_data: np.ndarray, y_val_data: np.ndarray) -> matplotlib.figure.Figure:
    
    fig = plt.figure()
    ax= plt.subplot()
    y_pred = clf.predict(X_val_data)
    cm = confusion_matrix(y_val_data, y_pred)
    hm = sns.heatmap(cm, annot=True, ax = ax, fmt='g')
    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    
    return fig

def generate_auc_roc_plot(clf: RandomForestClassifier, X_val_data: np.ndarray, y_val_data: np.ndarray) -> matplotlib.figure.Figure:
    ## Generate ROC-AUC Curve based on validation set
    fig, ax = plt.subplots(1)
    proba = clf.predict_proba(X_val_data).T[1]
    tpr, fpr, _ = roc_curve(y_val_data, proba)
    auc = roc_auc_score(y_val_data, proba)
    plt.plot(tpr, fpr)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.text(0.95, 0.01, u"%0.2f" % auc,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, weight='bold',
            fontsize=10)
    plt.suptitle('Validation AUC-ROC Curve')
    
    return fig

def generate_precision_recall_plot(clf: RandomForestClassifier, X_val_data: np.ndarray, y_val_data: np.ndarray) -> matplotlib.figure.Figure:
    fig, ax = plt.subplots(1)
    proba = clf.predict_proba(X_val_data).T[1]
    pred = clf.predict(X_val_data)
    precision, recall, _ = precision_recall_curve(y_val_data, proba)
    f1_val = f1_score(y_val_data, pred)
    plt.plot(recall, precision)
    ax.get_xaxis().set_visible(True)
    ax.get_yaxis().set_visible(True)
    ax.text(0.95, 0.01, u"%0.2f" % f1_val,
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes, weight='bold',
            fontsize=10)
    plt.suptitle('Validation Precision-Recall Curve')
    
    return fig

def main(args):
    print('Download Data...')
    # Load the pre-trained Mol2Vec vectors and download data
    model = load_model(os.path.join(args.data_dir,'model_300dim.pkl'))
    X, y = download_data(args.dev_mode, model)

    # split data into train and validation set, use stratified splitting because data is imbalanced
    X, X_val, y, y_val = split_data(X, y)
    
    kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    
    y_values = []
    predictions = []
    probas = []
    
    print('Start Training...')
    for train, test in kf.split(X, y):
        clf = RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=-1) # n_jobs == -1 -> use all available processors
        clf.fit(X[train], y[train])
        predictions.append(clf.predict(X[test]))
        probas.append(clf.predict_proba(X[test]).T[1]) # Probabilities for class 1
        y_values.append(y[test])
        
    print('Training Complete')
    aucs = [roc_auc_score(y, proba) for y, proba in zip(y_values, probas)]
    print(f'Mean AUC: {np.mean(aucs)}, Standard Devication AUC: {np.std(aucs)}')
    
    # Generate KFold training AUC-ROC curves and save to args.model_dir
    f, ((p1, p2, p3, p4)) = plt.subplots(1,4, squeeze=True, sharex=True, sharey=True, 
                                                    figsize=(12,3))

    for y,proba,ax in zip(y_values, probas, (p1,p2,p3,p4)):
        tpr, fpr, _ = roc_curve(y, proba)
        auc = roc_auc_score(y, proba)
        ax.plot(tpr, fpr)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.text(0.95, 0.01, u"%0.2f" % auc,
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, weight='bold',
                fontsize=10)
        plt.suptitle('KFold Train AUC-ROC')
    plt.savefig(os.path.join(args.model_dir, 'roc_auc.png'))
    
    # save the model
    filename = 'finalized_model.sav'
    joblib.dump(clf, os.path.join(args.model_dir,filename))
    
    # generate confusion matrix and save
    cm = generate_confusion_matrix_plot(clf, X_val, y_val)
    cm.savefig(os.path.join(args.model_dir, 'confusion_matrix.png'))
    
    # generate AUC-ROC curve for validation data
    auc_roc = generate_auc_roc_plot(clf, X_val, y_val)
    auc_roc.savefig(os.path.join(args.model_dir, 'auc_roc_validation.png'))
    
    # generate recall-precision curve for validation data
    rp_curve = generate_precision_recall_plot(clf, X_val, y_val)
    rp_curve.savefig(os.path.join(args.model_dir, 'recall_precision_validation.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-mode', type=str, default='False') 
    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    args = parser.parse_args()
    
    assert args.dev_mode.lower() == 'false' or args.dev_mode.lower() == 'true', '--dev-mode arg incorrect, should be "True" or "False"'

    main(args)
    
    
