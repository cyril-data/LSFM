from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import cluster, metrics
import seaborn as sns
import tensorflow as tf

from sklearn.preprocessing import KBinsDiscretizer
from math import pi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


def silhouettes_kmeans(XStd, max_it=3,  min_cluster=5, max_SF_cluster=15, plot=False):

    silhouettes = []

    for num_clusters in range(min_cluster, max_SF_cluster):
        # print("nb clusters tests : {}".format(num_clusters))
        for stab in range(max_it):
            #         n_init = 10
            cls = KMeans(n_clusters=num_clusters, random_state=None)
            cls.fit(XStd)
            silh = metrics.silhouette_score(XStd, cls.labels_)
            silhouettes.append(
                [num_clusters, stab, "n_init", silh, cls.labels_])
    if plot:
        sns.set(style="whitegrid")
        fig, axs = plt.subplots(1, 1, figsize=(5, 5))
        ax = sns.boxplot(x="K", y="silhouettes", data=df)

    return pd.DataFrame(silhouettes,
                        columns=["K", "random_state", "n_init", "silhouettes", 'labels'])


def init_r_discretizer(X, nb_r_class=10):
    #     X = np.array(reward_vect).reshape(-1,1)
    est = KBinsDiscretizer(
        n_bins=nb_r_class, encode='ordinal', strategy='uniform')
    est.fit(X)
    return est


def tranform_r_discretizer(X, r_discretizer):
    return r_discretizer.transform(X.values.reshape(-1, 1)).astype(int).reshape(-1)


def r_a_class_fc(r, a, r_class, a_class):
    r_class_norm = list(range(len(r_class)))
    r_norm = list(r_class).index(r)
    Nr = len(r_class_norm)
    Na = len(a_class)
    return a % Na * Nr + r_norm


def r_a_class_df_fc(X, r_discretizer, nb_action=4):

    df = X.loc[:, ["reward", "action"]]

    df["reward_discrete"] = tranform_r_discretizer(df["reward"], r_discretizer)

    r_class = np.sort(np.unique(df["reward_discrete"]))
    r_class
    r_inter = []
    for r_dis in r_class:
        #     df_inter = df.loc[df["reward_discrete"]  == r_dis]["reward" ]
        df_inter = df["reward"].loc[df["reward_discrete"] == r_dis]
        r_inter.append([df_inter.min(), df_inter.max()])

    a_class = list(range(nb_action))

    r_a_class_df = df.apply(lambda x: r_a_class_fc(x['reward_discrete'], x['action'], r_class, a_class),
                            axis=1).astype(int)
    df["r_a_class"] = r_a_class_df
    r_a_class = r_a_class_df.unique()

    r_a_inter = []
    for r_a_dis in r_a_class:
        df_inter = df.loc[df["r_a_class"] ==
                          r_a_dis].loc[:, ["reward", "action"]]
        r_a_inter.append([r_a_dis, df_inter["reward"].min(
        ), df_inter["reward"].max(), df_inter["action"].unique()[0]])

    r_a_inter_df = pd.DataFrame(data=r_a_inter, columns=[
                                "r_a_class", "r_min", "r_max", "action"])

    return r_a_class_df, r_a_inter_df


def reward_action_train_class(df, nb_r_class=10, nb_action=4, r_discretizer=None):
    if r_discretizer == None:
        r_discretizer = init_r_discretizer(
            df["reward"].values.reshape(-1, 1), nb_r_class)

    class_ra, r_a_class_index = r_a_class_df_fc(df, r_discretizer, nb_action=4)

    return class_ra, r_a_class_index, r_discretizer


def SF_sub_clustering(data_df, model_LSFM, r_a_class_index, max_SF_cluster=50):

    states_batch = list(data_df.values)

    best_random_state = 0

    clusters = []
    min_sample_nb = 200

    min_cluster = 2
    k = 0

    for i, ra_class in enumerate(r_a_class_index["r_a_class"]):
        set_ra_class = data_df.loc[data_df["r_a_class"] == ra_class]

        states_batch = list(set_ra_class.values)
        print("class {} : nb_indiv = {}, [reward min = {}, reward max = {}], action = {}".format(
            i,
            len(states_batch),
            r_a_class_index.loc[i, "r_min"],
            r_a_class_index.loc[i, "r_max"],
            r_a_class_index.loc[i, "action"],
        ))

        if len(states_batch) < min_sample_nb:
            cluster_labels = np.zeros((len(states_batch),), dtype=int)
            cluster_labels = cluster_labels + k
            clusters.append([list(set_ra_class.index), cluster_labels])
            k += 1
        else:
            states = tf.convert_to_tensor(
                np.array([val[0] for val in states_batch]))
            phis = model_LSFM(states)["phi"]

            df = silhouettes_kmeans(
                phis, max_it=1,  min_cluster=min_cluster, max_SF_cluster=max_SF_cluster)

            best_sil = df["silhouettes"].max()
            best_sil_id = df["silhouettes"].argmax()
            best_nb_cluster = df.loc[best_sil_id, "K"]

            cluster_labels = df.loc[best_sil_id, "labels"]

            print("best cluster : silhouette = {} for {} clusters".format(
                best_sil, best_nb_cluster))

            cluster_labels = cluster_labels + k

            clusters.append([list(set_ra_class.index), cluster_labels])

            k += best_nb_cluster

    return clusters


# ********************************************************
# How to use SF_clustering
# ----------------------------
#    for train :
# SF_clusters_df = SF_clustering(
#     buffer_train_df,model_LSFM, nb_r_class = 10 , nb_action = 4 , r_discretizer = None)

#     for test
# df = buffer_train_df
# nb_r_class = 10
# init_r_discretizer(df["reward"].values.reshape(-1,1), nb_r_class)
# df["reward_discrete"] = tranform_r_discretizer(df["reward" ], r_discretizer)
# SF_clusters_df = SF_clustering(
#     buffer_train_df,model_LSFM, nb_action = 4 , r_discretizer = r_discretizer)


def SF_clustering(data_df, model_LSFM, nb_r_class=10, nb_action=4, r_discretizer=None, max_SF_cluster=50):

    df = data_df

    df["r_a_class"], r_a_class_index, r_discretizer = reward_action_train_class(
        df, nb_r_class, nb_action, r_discretizer)

    SF_clusters = SF_sub_clustering(
        df, model_LSFM, r_a_class_index, max_SF_cluster)

    id_data_index_df = pd.DataFrame(data=SF_clusters, columns=["id", "label"])

    df["LSFM_cluster"] = df["r_a_class"]

    for k in range(len(SF_clusters)):
        df.loc[id_data_index_df.loc[k, "id"],
               "LSFM_cluster"] = id_data_index_df.loc[k, "label"]

    return df


def format_explode(data_df_input):
    df = data_df_input
    col_obs = ["cur_obs_"+str(k) for k in range(len(df["observation"][0]))]
    obs_serie = df[["observation", "LSFM_cluster", "action"]]
    col_obs_explode = col_obs * len(obs_serie)
    col_obs_explode = col_obs*len(obs_serie)
    obs_serie_explode = obs_serie.explode("observation", ignore_index=True)
    obs_serie_explode["stats_comp"] = col_obs_explode
    obs_serie_explode["observation"] = obs_serie_explode["observation"].astype(
        float)
    data_df = obs_serie_explode

    return data_df


def plot_ridge(data_df, title="", title_x="", hue_name="", pos_text_x=1.05, pos_text_y=1.5, save_fig="plot_ridge.jpg"):

    row_name = 'stats_comp'
    map_name = 'observation'

    col_obs = list(data_df[row_name].unique())
    dict_name = {}
    for index, value in enumerate(col_obs):
        dict_name[index+1] = value
    dict_name

    # we generate a color palette with Seaborn.color_palette()
    pal = sns.color_palette(palette='coolwarm', n_colors=12)
    # in the sns.FacetGrid class, the 'hue' argument is the one that is the one that will be represented by colors with 'palette'
    if hue_name == "":
        g = sns.FacetGrid(data_df, row=row_name, aspect=15,
                          height=0.75, palette=pal)
    else:
        g = sns.FacetGrid(data_df, row=row_name, hue=hue_name,
                          aspect=15, height=0.75, palette=pal)

    # then we add the densities kdeplots for each month
    g.map(sns.kdeplot, map_name,
          bw_adjust=1, clip_on=False,
          fill=True, alpha=1, linewidth=1.5)

    # here we add a white line that represents the contour of each kdeplot
    g.map(sns.kdeplot, map_name,
          bw_adjust=1, clip_on=False,
          color="w", lw=2)

    # here we add a horizontal line for each plot
    g.map(plt.axhline, y=0,
          lw=2, clip_on=False)

    # we loop over the FacetGrid figure axes (g.axes.flat) and add the month as text with the right color
    # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
    for i, ax in enumerate(g.axes.flat):
        ax.text(pos_text_x, pos_text_y, dict_name[i+1],
                fontweight='bold', fontsize=15,
                color=ax.lines[-1].get_color())

    # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.3)

    # eventually we remove axes titles, yticks and spines
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.setp(ax.get_xticklabels(), fontsize=15, fontweight='bold')
    plt.xlabel(title_x, fontweight='bold', fontsize=15)
    g.fig.suptitle(title,
                   ha='right',
                   fontsize=20,
                   fontweight=20)

    g.set(xlim=(0, 1.0))

    plt.savefig(save_fig)
    # plt.show()

    return plt


# ********************************************************
# How to use split_ridge_plot
# ----------------------------
#
# split_analysis = [
#     list(range(0,12)),
#     list(range(12,24)),
#     list(range(24,36))
# ]
# split_analysis_col = []
# for i, comp_list in enumerate(split_analysis) :
#     split_analysis_col.append(["cur_obs_"+str(k) for k in comp_list])
# split_analysis_col

# split_ridge_plot(SF_clusters_df,
# #            split = [split_analysis_col],
#            title = 'States components densities',
#            title_x = 'States components values')
# ********************************************************

def split_ridge_plot(data_df_input,
                     split=[],
                     title="",
                     title_x="",
                     hue_name="",
                     pos_text_x=1.05,
                     pos_text_y=1.5,
                     save_fig="plot_ridge.jpg"):

    #   apply format plot_ridge
    df = data_df_input
    col_obs = ["cur_obs_"+str(k) for k in range(len(df["observation"][0]))]
    obs_serie = df[["observation", "LSFM_cluster", "action"]]
    col_obs_explode = col_obs * len(obs_serie)
    col_obs_explode = col_obs*len(obs_serie)
    obs_serie_explode = obs_serie.explode("observation", ignore_index=True)
    obs_serie_explode["stats_comp"] = col_obs_explode
    obs_serie_explode["observation"] = obs_serie_explode["observation"].astype(
        float)
    data_df = format_explode(data_df_input)

    if split == []:
        plot_ridge(data_df,
                   title=title,
                   title_x=title_x,
                   pos_text_y=0.,
                   save_fig=save_fig)

    else:
        for i, comp_analysis in enumerate(split_analysis_col):
            analysis = []
            for comp in comp_analysis:
                analysis.append(data_df.loc[data_df['stats_comp'] == comp])
            analysis = pd.concat(analysis)

            plot_ridge(analysis,
                       title=title,
                       title_x=title_x,
                       pos_text_y=0.,
                       save_fig=save_fig)


def select_col_by_list(data_df, col, col_list):
    analysis = []
    for comp in col_list:
        if col not in list(data_df.columns):
            print(
                "error in select_col_by_list : {} is not in data_df.columns".format(col))
            break

        if comp not in data_df[col].unique():
            print("error in select_col_by_list : {} is not in data_df[{}] values".format(
                comp, col))
            break

        analysis.append(data_df.loc[data_df[col] == comp])

    analysis = pd.concat(analysis)
    return analysis


# ********************************************************
# How to use spider_cls
# ----------------------------
#
# filtre_action = [0]
# filtre_comp_int = [0,1,2,3,5,6,7,8,9]
# spider_cls_mean(SF_clusters_df, filtre_action, filtre_comp_int)


def spider_cls(df, filtre_action=[], filtre_comp_int=[], reduce="mean", save_fig="spider_cls.jpg"):

    col_obs = ["cur_obs_"+str(k) for k in range(len(df["observation"][0]))]
    obs_df = pd.DataFrame(df["observation"].to_list(), columns=col_obs)
    obs_df["LSFM_cluster"] = df["LSFM_cluster"]
    obs_df["action"] = df["action"]
    df = obs_df

    #  df filter with filtre_action :
    if filtre_action != []:
        df = select_col_by_list(df, "action", filtre_action)

    # df filter with state filtre_comp_int
    if filtre_comp_int != []:
        comp_list = [name for name in list(df.columns) if "cur_obs" in name]
        filtre_comp = [name for i, name in enumerate(
            comp_list) if "cur_obs" in name and int(name.split("_")[2]) in filtre_comp_int]
        col_remove = [elem for elem in comp_list if elem not in filtre_comp]
        df = df.drop(col_remove, axis=1)

    fig = plt.figure(figsize=(7, 7))

    if reduce == "mean":
        df = df.groupby(['LSFM_cluster']).mean()
    elif reduce == "var":
        df = df.groupby(['LSFM_cluster']).var()
    else:
        return "error in spider_cls : reduce != 'mean' or 'var' "

    df["LSFM_cluster"] = df.index

    group = ["LSFM_cluster", "action"]
    y_lim_max = 1.

    print("df", df)

    print("filtre_comp_int", filtre_comp_int)
    # ------- PART 1: Create background

    # number of variable
    categories = [int(name.split("_")[2]) for i, name in enumerate(
        list(df.columns)) if "cur_obs" in name]
    if filtre_comp_int != []:
        categories = filtre_comp_int
    N = len(categories)

    print("categories", categories)
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.5, 0.8], ["0.2", "0.5", "0.8"], color="grey", size=7)
    plt.ylim(0, y_lim_max)

    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    print("categories", categories)

    for i in range(len(df)):
        values = df.iloc[i].drop(group).values.flatten().tolist()

        values += values[:1]
        print("values", df.iloc[i].drop(group).values.shape, values)
        print("angles", angles)
        ax.plot(angles, values, linewidth=1,
                linestyle='solid', label="cls"+str(i))
        ax.fill(angles, values, 'b', alpha=0.1)

    plt.xlabel("clusters mean on state components",
               fontweight='bold', fontsize=15)
    plt.legend(loc='upper right', facecolor="white",
               ncol=4, bbox_to_anchor=(2, 1))

    # plt.show()
    plt.savefig(save_fig)


def plot_spiders(SF_clusters_df, nb_action=4, fold_result=""):

    filtre_action = []
    filtre_comp_int = []

    spider_cls(
        SF_clusters_df,
        filtre_action,
        filtre_comp_int,
        save_fig=fold_result+"_spider.jpg")

    for a in range(nb_action):
        spider_cls(
            SF_clusters_df,
            [a],
            filtre_comp_int,
            save_fig=fold_result+"_spider_a" + str(a)+".jpg")

    split_analysis = [
        list(range(0, 12)),
        list(range(12, 24)),
        list(range(24, 36))
    ]
    for filtre_comp_int in split_analysis:
        spider_cls(
            SF_clusters_df,
            filtre_action,
            filtre_comp_int,
            save_fig=fold_result+"_spider_comp" + str(filtre_comp_int) + ".jpg")
