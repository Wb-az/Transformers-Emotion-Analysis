import os
import numpy as np
from natsort import natsorted
from mlxtend.evaluate import mcnemar_table, mcnemar
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import pandas as pd
import matplotlib.pyplot as plt


pred_dir = 'predictions'
eval_dir = 'metrics'

def save_file(predictions, path, file_name):
    """
    :param predictions: a tensor with predictions
    :param path: path to extract the predition
    :param file_name: name to save the predictions
    :return: compresed numpy file
    """
    path = os.path.join(path, file_name)
    return np.savez_compressed(path, predictions)


def load_npz_file(path):
    """
    Parameters
    ----------
    path : string
        path to predictions
    Returns
    -------
    result: array
        loads the data stored
    """
    dict_data = np.load(path, allow_pickle=True)
    return dict_data['arr_0']


def model_to_path(model, base_dir):
    """
    
    Parameters
    ----------
    model : string
        model to extract metrics
    base_dir : string
        base darectpry to create the path
    Returns
    -------
    result: string
        a path to the csv file
    """
    
    return os.path.join(base_dir, model)
    

def get_metrics(models, base_dir):
    """
    Parameters
    ----------
    models : list
        a list wit the name of the models
    base_dir : string
        string with the base path
    Returns
    -------
    df : dataframe
        a dataframe with the metrics for all experiments

    """
   
    df = pd.DataFrame()

    for i,  model in enumerate(models):
        df_temp = pd.read_csv(model_to_path(model, base_dir))
        df_temp = df_temp.iloc[0]        
        df = df.append(df_temp)
        
    df.set_index('exp', inplace=True)
        
    return df


def mcnemar_comparison(target, pred1, pred2):
    """
    Parameters
    ----------
    target : numpy array
        true labels
    model1 : numpy array
        predictions from model1
    model2 : numpy array
        predictions from model2

    Returns
    -------
    result: numeric (float)
        chi-squared statistic and p-value

    """
    # Create the contingency table
    c_table = mcnemar_table(target, pred1, pred2)
    # Computes the p-value as chi2 statistic
    chi2, p = mcnemar(ary=c_table, corrected=True)
    
    return chi2, p



def df_to_latex(fname, caption, label, df):
    """
    Parameters
    ----------
    fname : string
        name  to save the file
    caption : string
        caption for the table
    label : string
        label for the table
    df : dataframe
        dataframe to convert into text
    Returns
    -------
    Dataframe in LaTeX format .text

    """
    n_columns = len(df.keys()) + 1
    fname = fname + '.tex'
    with open(os.path.join(os.getcwd(), fname), 'w') as tf:
        tf.write(df.to_latex(caption=caption,
                                           label=label, escape=False,
                                           column_format='l' * n_columns))
        
    return fname
        
 
    
if __name__ == '__main__':
    
    y_true = load_npz_file('true_labels/true_labels.npz')
    pred_list = natsorted(os.listdir('predictions'))
    pred_list = [m for m in pred_list if m != '.DS_Store']
    metrics_list = natsorted(os.listdir('metrics'))
    
    labels= ['Exp-' + str(x+1).zfill(2) for x in range (len(pred_list))]
    
    arra = np.zeros((len(y_true), len(pred_list)))

    
    for i, pred in enumerate(pred_list):
        y_pred = load_npz_file(os.path.join(pred_dir, pred))
        arra[:,i] = y_pred

    stats, p = friedmanchisquare(*arra.T)
  
    
    if p < 0.05:
        print(p)
        print('p-value < 0.05, reject H0')
        
        mn = np.zeros((len(pred_list), len(pred_list)))

        for i, pred in enumerate(pred_list):
            y1 = load_npz_file(os.path.join(pred_dir, pred))
        
            for ii in range (0, len(pred_list)):
                
                y2 = load_npz_file(os.path.join(pred_dir, pred_list[ii]))
    
                ch1, p = mcnemar_comparison(y_true, y1, y2)
                
                mn[i,ii] = p
           
        fig = plt.figure(figsize= (8,20))
        cmap = ['1', '#FADCDC',  '#08306b',  '#4292c6', '#c6dbef']
        heatmap_args = {'cmap':cmap, 'linewidths': 0.2, 'linecolor': '0.2', 
                     'clip_on': True, 'square': True}

        sp.sign_plot(mn, labels, **heatmap_args) 
        plt.savefig(os.path.join('./plots', 'mcnemar.png'), bbox_inches='tight',
                 format='png', dpi=2000)

    
    

    
