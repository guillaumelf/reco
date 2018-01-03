# -*- coding: utf-8 -*-
__author__ = 'Guillaume Le Floch'

### Import des librairies utiles
################################

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.preprocessing import MaxAbsScaler
from concurrent.futures import ThreadPoolExecutor
import operator

### Définition de fonctions locales
###################################

def sorted_dict(d):
    return sorted(d.items(), key=lambda t: t[1], reverse=True)

def export_result(lst_results,df,filename,indicator,nb_top=30):
    top = lst_results[:nb_top]
    ind = [elem[0] for elem in top]
    score = [elem[1] for elem in top]
    subset_df = df[df['id'].isin(ind)]
    content = [('id',ind),('score',np.round(score,decimals=6))]
    res = pd.DataFrame.from_items(content)
    final = pd.merge(subset_df, res, how='inner', on='id')
    results_df = final.sort_values(by=['score'],ascending=False)
    print('Les résultats sont les suivants pour {} :'.format(indicator))
    print('')
    print(results_df)
    print('')
    print('##################################################')
    print('')
    results_df.to_csv(filename,sep=';',header=True,decimal='.',encoding='utf-8',index=False)

def score_normalizer(method):
    ind = [elem[0] for elem in method]
    score = [elem[1] for elem in method]
    normalized_score = scaler.fit_transform(np.array(score).reshape(-1, 1))
    for i in range(len(ind)):
        try:
            dico_scores[ind[i]] += float(normalized_score[i])
        except KeyError:
            dico_scores[ind[i]] = float(normalized_score[i])
            
def generic_common_neighbors(g, u, v):
    list_common_neighbors = []
    for w in g.nodes():
        if (g.has_edge(u, w) and g.has_edge(w, v)) or (g.has_edge(v, w) and g.has_edge(w, u)):
            list_common_neighbors.append(w)
    return list_common_neighbors


def generic_adamic_adar(g, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(g)

    def predict(u, v):
        return sum([1. / np.log(g.degree(w)) for w in generic_common_neighbors(g, u, v)])

    return [(u, v, predict(u, v)) for u, v in ebunch]

def select_prediction(id_product):
    result = [(pred[1],pred[2]) for pred in predictions if pred[0] == id_product]
    return sorted(result, key=operator.itemgetter(1), reverse=True)
           
def make_results(prod_list):
    return {prod: select_prediction(prod) for prod in id_products}

### Corps principal du programme
################################

# Importation du graphe et des méta données que l'on va stocker dans un dataframe
G=nx.read_edgelist("dvd.txt",create_using=nx.DiGraph())
fig=plt.figure(figsize=(15,10))
nx.draw_networkx(G, with_labels=True)
fig.savefig('Images/graphe.jpg')
plt.show()
#G=nx.read_edgelist("Amazon0601.txt",create_using=nx.DiGraph())
with open('amazon-meta.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
    new_lines = [line.strip() for line in lines] # on enlève les espaces
    new_lines = [line for line in new_lines if not (line == '')] # puis les lignes vides

for ix in range(14456151): # on procède à un traitement pour éliminer les identifiants n'ayant pas de description
    if (new_lines[ix].startswith('Id')) and (new_lines[(ix+2)].startswith('title')==False):
        new_lines.pop(ix)

ids = [line.split(' ')[-1] for line in new_lines if line.startswith('Id')]
title = [line.split(' ')[-1] for line in new_lines if line.startswith('title')]
title = [re.sub(r"\(",r"",exp) for exp in title]
title = [re.sub(r"\)",r"",exp) for exp in title]
content = [('id',ids),('title',title)]
df = pd.DataFrame.from_items(content)

# On calcule les degrés entrants et on affiche les résultats

dc = sorted_dict(nx.in_degree_centrality(G))
export_result(dc,df,'in_degree_centrality.csv','le degré de centralité entrant')

# On calcule la centralité de proximité

cc = sorted_dict(nx.closeness_centrality(G,reverse=True))
export_result(cc,df,'closeness_centrality.csv','la centralité de proximité')

# Puis la centralité d'intermédiarité

bc = sorted_dict(nx.betweenness_centrality(G))
export_result(bc,df,'betweenness_centrality.csv',"la centralité d'intermédiarité")


# Puis la centralité de vecteur propre

ec = sorted_dict(nx.eigenvector_centrality(G,max_iter=200))
export_result(ec,df,'eigenvector_centrality.csv','la centralité de vecteur propre')

# Et enfin l'algorithme PageRank

pr = sorted_dict(nx.pagerank(G,alpha=0.85))
export_result(pr,df,'pagerank.csv','la méthode PageRank')

###################################################################################

# Sélection des DVD sur lesquels on va effectuer de la prédiction

scaler = MaxAbsScaler() # On normalise les scores par rapport au max
methods = [dc,cc,bc,ec,pr]
dico_scores = {}
e = ThreadPoolExecutor()            
e.map(score_normalizer,methods)
export_result(sorted_dict(dico_scores),df,'top30.csv','les scores cumulés normalisés des différentes méthodes',30)

###################################################################################
### Partie prédiction de liens

# Extraction des identifiants des produits qui nous intéressent

top10 = sorted_dict(dico_scores)[:10]
id_products = [elem[0] for elem in top10]
dico_prod = {ident:df.get_value(df[df.id == ident].index[0],'title') for ident in id_products}

# Extraction du sous-graphe contenant ces produits et ceux auxquels ils sont reliés

u = G.to_undirected()
nodes = []
for id_prod in id_products:
    nodes += list(nx.shortest_path(u,id_prod).keys())
    neighbors = u.neighbors(id_prod)
    for n in neighbors:
        nodes += list(nx.shortest_path(u,n).keys())
s = G.subgraph(nodes)

# Prédictions de nouveaux liens avec l'index adamic adar (avec une modification de la fonction qui permet de l'appliquer
# à un graphe orienté)

predictions = generic_adamic_adar(s)
pred_clean = make_results(id_products)

# On extrait les 5 produits pour lesquels l'index est le plus fort

for k,v in pred_clean.items():
    export_result(v,df,dico_prod[k]+'_pred.csv','la prédictions de liens du DVD {}'.format(dico_prod[k]),5)