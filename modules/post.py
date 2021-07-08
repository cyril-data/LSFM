import tensorflow as tf
from tensorflow import keras
import pandas as pd

def norm(vect) : 
    return tf.keras.backend.get_value(tf.norm(vect, ord='euclidean'))



def losses(col, y_pred_df, y_true_df) : 

    ypred = y_pred_df[col]
    ytrue = y_true_df[col]
    
    if (ytrue.empty or ypred.empty) : return [None, None,  col]
    
    if (ytrue.dtypes == "float64" or ytrue.dtypes == "float32" or ytrue.dtypes == float): 
        
        MSE = tf.keras.backend.get_value(keras.losses.mean_squared_error(ypred, ytrue))
        MAE = tf.keras.backend.get_value(keras.losses.mean_absolute_error(ypred, ytrue))
#         r2 = r2_score(ypred, ytrue)
    elif (ytrue.dtypes == "object"): 
        
        df = ytrue - ypred
        df = ytrue - ypred
        MSE =  df.apply(norm).sum()/len(df)
        MAE = None
    
    else : 
        return [None, None,  col]
        
    return [MSE, MAE,  col]

def global_loss(y_pred_df, y_true_df) : 
    errors = []
    for col in y_pred_df.columns : 
        errors.append(losses(col, y_pred_df, y_true_df))

    errors_df = pd.DataFrame(errors, columns=["MSE", "MAE", "error"])
        
    return errors_df

def losses_on_positive_rewards(col, y_pred_df, y_true_df, thresold = 0.5, nb_actions = 4) : 
    
    losses_action = []
    df = y_true_df
    
    for action in range(nb_actions):
        id_action = list(df.loc[df["reward_one-step"]>thresold].loc[df["action"]==action].index)

        
        loss = losses(col, y_pred_df.loc[id_action], y_true_df.loc[id_action])

        loss.insert(0, action)

        losses_action.append(loss)

    loss_action_df = pd.DataFrame(losses_action, columns=["action", "MSE", "MAE", "error"])
        
    return loss_action_df

def losses_on_positive_rewards_global(y_pred_df, y_true_df, thresold = 0.5, nb_actions = 4) : 
    data = pd.DataFrame()
    for col in list(y_pred_df.columns) : 
        loss = losses_on_positive_rewards(col, y_pred_df, y_true_df, thresold = 0.5, nb_actions = 4) 
        data = pd.concat([data, loss])
    return data