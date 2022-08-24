import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchmetrics import AUROC
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
    matthews_corrcoef, balanced_accuracy_score, precision_score, recall_score, f1_score


class Metrics(object):
    """
    Computes classification metrics for the test subset: Acc, BA, F1, F2,
    MCC, confusion matrix, classification report sensitivity, specificity,
    precision and auroc.
    """

    def __init__(self, y_true, y_pred, tensor_prob, classes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.prob = tensor_prob
        self.classes = sorted(classes)
        self.cf_matrix = confusion_matrix(self.y_true, self.y_pred)

    def acc(self):
        return accuracy_score(self.y_true, self.y_pred)
        
    def mcc(self):
        return matthews_corrcoef(self.y_true, self.y_pred)

    def balanced_acc(self):
        return balanced_accuracy_score(self.y_true, self.y_pred)

    def classification_rep(self):
        target_names = sorted(self.classes)
        return classification_report(self.y_true, self.y_pred, target_names=target_names, digits=3)

    def cmatrix(self):
        df_cm = pd.DataFrame(self.cf_matrix, index=sorted(self.classes),
                             columns=sorted(self.classes))
        return self.show_confusion_matrix(df_cm)

    def macro_average(self):
        macro_pres = precision_score(self.y_true, self.y_pred, average='macro')
        macro_rec = recall_score(self.y_true, self.y_pred, average='macro')
        macro_f1 = f1_score(self.y_true, self.y_pred, average='macro')
        pres_class = precision_score(self.y_true, self.y_pred, average=None)
        rec_class = recall_score(self.y_true, self.y_pred, average=None)
        f1_class = f1_score(self.y_true, self.y_pred, average=None)
        return macro_pres, macro_rec, macro_f1, pres_class, rec_class, f1_class 

    @staticmethod
    def show_confusion_matrix(cm):
        hmap = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=90, ha='right')
        plt.ylabel('True emotion')
        plt.xlabel('Predicted emotion')

    def auroc(self):
        probs = self.prob.view(-1, len(self.classes))
        # true = torch.as_tensor(self.y_true)
        auroc = AUROC(num_classes=len(self.classes))
        return auroc(probs.to('cpu'), self.y_true.to('cpu'))

def train_metrics(**train_dict):
    """
    :param train_dict: a dictionary with the accuracy and loss logs for training and validation
    :return:  max accuracies and minimum loss for validation and train datasets
    """
    max_acc = max(train_dict['train_acc'])
    min_loss = min(train_dict['val_loss'])
    val_acc = max(train_dict['val_acc'])
    min_val_loss = min(train_dict['val_loss'])

    return max_acc, min_loss, val_acc, min_val_loss