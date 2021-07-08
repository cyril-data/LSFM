import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


def norm(vect) : 
    return tf.keras.backend.get_value(tf.norm(vect, ord='euclidean'))


def classif(y_true, y_pred, dt_string, label = None) : 

#     reward_classes = [[-np.inf,parser[0]]] + [[parser[k],parser[k+1]]  for k in range(len(parser)-1)] + [[parser[-1], np.inf]]
    classes = y_true.unique()
    reward_classes = classes
    
    if label != None : reward_classes = label
        
    conf = multilabel_confusion_matrix(y_true, y_pred, labels=classes)
    conf

    error_classif = []

    for i,conf_bin in enumerate(conf) : 
        TN = conf_bin[0][0]
        FP = conf_bin[0][1]
        FN = conf_bin[1][0]
        TP = conf_bin[1][1]
        labels = np.array([["True Neg : {} \n or {:2.1f} % ".format(TN,100*TN / np.sum(TN + FP + FN + TP)),
                            "False Pos : {} \n or {:2.1f} % ".format(FP,100*FP / (TN + FP + FN + TP))],
                           ["False Neg : {} \n or {:2.1f} % ".format(FN,100*FN / (TN + FP + FN + TP)),
                            "True Pos : {} \n or {:2.1f} % ".format(TP,100*TP / (TN + FP + FN + TP))     ]])
        recall = TP / (FN + TP)
        precision = TP / (FP + TP)
        f1 = 2* recall * precision / (recall + precision)

        error_classif.append([recall, reward_classes[classes[i]] , "recall"])
        error_classif.append([precision, reward_classes[classes[i]] , "precision"])
        error_classif.append([f1, reward_classes[classes[i]] , "f1"])


        fig, ax = plt.subplots()
        sns.heatmap(conf_bin ,annot=labels, fmt="" , cmap='Blues').set(
            title='confusion matrix for rewards in {}'.format(reward_classes[classes[i]]))
    #         plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')


    # sns.heatmap(cf_matrix, annot=labels, fmt=‘’, cmap='Blues')
        plt.savefig(dt_string+"confusion_"+str(i)+".jpg")
    error_classif_df = pd.DataFrame(error_classif, columns = ["error", "reward_classes", "type"])
    error_classif_df["reward_classes"] = error_classif_df["reward_classes"].astype(str)
    return error_classif_df

def losses_reg(col, y_pred_df, y_true_df) : 

    ypred = y_pred_df[col]
    ytrue = y_true_df[col]    
    
    MSE, MAE = [None, None]
    
    if (ytrue.empty or ypred.empty) : return [MSE, MAE, col]
    
    if (ytrue.dtypes == "float64" or ytrue.dtypes == "float32" or ytrue.dtypes == float): 
        
        MSE = tf.keras.backend.get_value(keras.losses.mean_squared_error(ypred, ytrue))
        MAE = tf.keras.backend.get_value(keras.losses.mean_absolute_error(ypred, ytrue))
#         r2 = r2_score(ypred, ytrue)
    elif (ytrue.dtypes == "object"): 
        
        df = ytrue - ypred
        df = ytrue - ypred
        MSE =  df.apply(norm).sum()/len(df)
        MAE = None
        
    return [MSE, MAE, col]

def global_loss_reg(y_pred_df, y_true_df) : 
    errors = []
    for col in y_pred_df.columns : 
        errors.append(losses_reg(col, y_pred_df, y_true_df))

    errors_df = pd.DataFrame(errors, columns=["MSE", "MAE", "error"])
        
    return errors_df



def losses_on_reward_class(col, y_pred_df, y_true_df, reward_class = 2, nb_actions = 4) : 
    
    losses_action = []
    df = y_true_df
    
    for action in range(nb_actions):
        id_action = list(df.loc[df["reward_one-step"]==reward_class].loc[
                df["action"]==action].index)            
        
        loss = losses_reg(col, y_pred_df.loc[id_action], y_true_df.loc[id_action])

        loss.insert(0, action)

        losses_action.append(loss)

    loss_action_df = pd.DataFrame(losses_action, columns=["action", "MSE", "MAE", "error"])
        
    return loss_action_df

def losses_on_rewards_global(cols, y_pred_df, y_true_df, reward_class = 2, nb_actions = 4) : 
    data = pd.DataFrame()
    for col in list(cols) : 
        loss = losses_on_reward_class(
            col, y_pred_df, y_true_df, reward_class = reward_class, nb_actions = nb_actions) 
        data = pd.concat([data, loss])
    return data