from numpy import where
from collections import Counter
from sklearn import manifold
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, plot_confusion_matrix
import numpy as np
import seaborn as sns
sns.set()


def plot_data(
        data_plot,
        x="cum_step",
        y=[
            "Avg_loss",
            "Avg_loss_r",
            "Avg_loss_psi",
            "Avg_loss_phip1",
            "exploration_ratio",
            "eigen_exploration"],
        hue="eigen_opt",
        style="eigen",
        datafile="online_eigendiscovery.jpg"):

    fig, axs = plt.subplots(2, 3, figsize=(25, 15))

    # for option in options :
    hue = "eigen_opt"
    x = "cum_step"
    style = "eigen"

    sns.lineplot(x=x, y="Avg_loss",  hue=hue,
                 style=style, data=data_plot, ax=axs[0, 0])
    sns.lineplot(x=x, y="Avg_loss_r",  hue=hue,
                 style=style, data=data_plot, ax=axs[0, 1])
    sns.lineplot(x=x, y="Avg_loss_psi",  hue=hue,
                 style=style, data=data_plot, ax=axs[0, 2])
    sns.lineplot(x=x, y="Avg_loss_phip1",  hue=hue,
                 style=style, data=data_plot, ax=axs[1, 0])
    sns.lineplot(x=x, y="exploration_ratio",  hue=hue,
                 style=style, data=data_plot, ax=axs[1, 1])
    sns.lineplot(x=x, y="eigen_exploration",  hue=hue,
                 style=style, data=data_plot, ax=axs[1, 2])

    plt.savefig(datafile)
    plt.close()


def norm(vect):
    return tf.keras.backend.get_value(tf.norm(vect, ord='euclidean'))


def classif(y_true, y_pred, path, label=None):

    #     reward_classes = [[-np.inf,parser[0]]] + [[parser[k],parser[k+1]]  for k in range(len(parser)-1)] + [[parser[-1], np.inf]]
    classes = y_true.unique()
    reward_classes = classes

    if label != None:
        reward_classes = label

    conf = multilabel_confusion_matrix(y_true, y_pred, labels=classes)
    conf

    error_classif = []

    for i, conf_bin in enumerate(conf):
        TN = conf_bin[0][0]
        FP = conf_bin[0][1]
        FN = conf_bin[1][0]
        TP = conf_bin[1][1]
        labels = np.array([["True Neg : {} \n or {:2.1f} % ".format(TN, 100*TN / np.sum(TN + FP + FN + TP)),
                            "False Pos : {} \n or {:2.1f} % ".format(FP, 100*FP / (TN + FP + FN + TP))],
                           ["False Neg : {} \n or {:2.1f} % ".format(FN, 100*FN / (TN + FP + FN + TP)),
                            "True Pos : {} \n or {:2.1f} % ".format(TP, 100*TP / (TN + FP + FN + TP))]])
        recall = TP / (FN + TP)
        precision = TP / (FP + TP)
        f1 = 2 * recall * precision / (recall + precision)

        error_classif.append([recall, reward_classes[classes[i]], "recall"])
        error_classif.append(
            [precision, reward_classes[classes[i]], "precision"])
        error_classif.append([f1, reward_classes[classes[i]], "f1"])

        fig, ax = plt.subplots()
        sns.heatmap(conf_bin, annot=labels, fmt="", cmap='Blues').set(
            title='confusion matrix for rewards in {}'.format(reward_classes[classes[i]]))
        plt.savefig(path+"confusion_"+str(i)+".jpg")
        plt.close()

    error_classif_df = pd.DataFrame(
        error_classif, columns=["error", "reward_classes", "type"])
    error_classif_df["reward_classes"] = error_classif_df["reward_classes"].astype(
        str)
    return error_classif_df


def losses_reg(col, y_pred_df, y_true_df):

    ypred = y_pred_df[col]
    ytrue = y_true_df[col]

    MSE, MAE = [None, None]

    if (ytrue.empty or ypred.empty):
        return [MSE, MAE, col]

    if (ytrue.dtypes == "float64" or ytrue.dtypes == "float32" or ytrue.dtypes == float):

        MSE = tf.keras.backend.get_value(
            keras.losses.mean_squared_error(ypred, ytrue))
        MAE = tf.keras.backend.get_value(
            keras.losses.mean_absolute_error(ypred, ytrue))
#         r2 = r2_score(ypred, ytrue)
    elif (ytrue.dtypes == "object"):

        df = ytrue - ypred
        df = ytrue - ypred
        MSE = df.apply(norm).sum()/len(df)
        MAE = None

    return [MSE, MAE, col]


def global_loss_reg(y_pred_df, y_true_df):
    errors = []
    for col in y_pred_df.columns:
        errors.append(losses_reg(col, y_pred_df, y_true_df))

    errors_df = pd.DataFrame(errors, columns=["MSE", "MAE", "error"])

    return errors_df


def losses_on_reward_class(col, y_pred_df, y_true_df, reward_class=2, nb_action=4):

    losses_action = []
    df = y_true_df

    for action in range(nb_action):
        id_action = list(df.loc[df["reward_one-step"] == reward_class].loc[
            df["action"] == action].index)

        loss = losses_reg(
            col, y_pred_df.loc[id_action], y_true_df.loc[id_action])

        loss.insert(0, action)

        losses_action.append(loss)

    loss_action_df = pd.DataFrame(losses_action, columns=[
                                  "action", "MSE", "MAE", "error"])

    return loss_action_df


def losses_on_rewards_global(cols, y_pred_df, y_true_df, reward_class=2, nb_action=4):
    data = pd.DataFrame()
    for col in list(cols):
        loss = losses_on_reward_class(
            col, y_pred_df, y_true_df, reward_class=reward_class, nb_action=nb_action)
        data = pd.concat([data, loss])
    return data


# plot One-hot states as integer with actions
def plot_state_int_action(buffer_df, save_fig):

    a = np.array(buffer_df["observation"].values.tolist())
    states_int = np.argmax(a, axis=1)
    action_int = np.array(buffer_df["action"].values.tolist())
    action_int

    reward_S = buffer_df["reward"]
    y_float = pd.Series(np.array(reward_S.values.tolist()))
    y_class = y_float.apply(lambda x: discretize(x, reward_parser))

    y = y_class

    # Generate and plot a synthetic imbalanced classification dataset
    from collections import Counter
    from sklearn.datasets import make_classification
    from numpy import where
    # summarize class distribution
    counter = Counter(y)
    print(counter)
    # scatter plot of examples by class label
    fig, ax = plt.subplots(figsize=(10, 7))
    for label, _ in counter.items():
        row_ix = where(y == label)[0]
        plt.scatter(states_int[row_ix], action_int[row_ix], label=str(label))
    plt.xlabel("states")
    plt.ylabel("actions")

    yint = []
    locs, labels = plt.yticks()
    for each in locs:
        yint.append(int(each))
    plt.yticks(yint)

    plt.legend()
    plt.savefig(save_fig)
    plt.close()
# plot 2d embelled TSNE on states-actions-newstates with class reward


def plot_2d_embelled(XStd, y_class, save_fig):
    n_components = 2
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'xy'}]])
    perplexity = 10
    embedding = manifold.TSNE(
        n_components=n_components, perplexity=perplexity, init="pca", random_state=0)
    # --- Visu 2d
    X_transformed = embedding.fit_transform(XStd)
    X = X_transformed
    y = y_class

    counter = Counter(y)
    print(counter)

    for label, _ in counter.items():
        row_ix = where(y == label)[0]

        plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
    plt.legend()
    plt.savefig(save_fig)
    plt.close()


def plot_error(df, folder):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    sns.lineplot(x="cum_step", y="Avg_loss", data=df, ax=axs[0, 0])
    sns.lineplot(x="cum_step", y="Avg_loss_r", data=df, ax=axs[0, 1])
    sns.lineplot(x="cum_step", y="Avg_loss_N", data=df, ax=axs[0, 2])
    sns.lineplot(x="cum_step", y="Avg_loss_psi", data=df, ax=axs[1, 0])
    plt.savefig(folder+"_Loss_LSFM.jpg")
    plt.close()


def plot_classif_reward_error(error_classif_df, folder):
    fig, ax = plt.subplots()
    sns.barplot(x="reward_classes", y="error", hue="type", data=error_classif_df).set(
        title='Test classif reward errors', xlabel="")
    plt.savefig(folder+"_classif_reward_error.jpg")
    plt.close()
