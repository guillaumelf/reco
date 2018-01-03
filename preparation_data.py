# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 20:25:26 2017

@author: Guillaume
"""

### Import des librairies utiles
################################

import networkx as nx
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor

### Définition de fonctions locales
###################################

def sorted_dict(d):
    return sorted(d.items(), key=lambda t: t[1], reverse=True)

def is_dvd(line):
    if (str(line.split()[0]) in id_dvd) and (str(line.split()[1]) in id_dvd):
        dvd.write(line)

### Corps principal du programme
################################

# Importation du graphe et des méta données que l'on va stocker dans un dataframe
G=nx.read_edgelist("Amazon0601.txt",create_using=nx.DiGraph())
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
group = [line.split(' ')[-1] for line in new_lines if line.startswith('group')]
content = [('id',ids),('title',title),('group',group)]
df = pd.DataFrame.from_items(content)
df = df[~df.title.isin(['II','Other','1','2','3','4','5','6','7','8','9'])]
print(df.groupby('group').count())
# On va extraire les dvd seulement, par leur identifiant
id_dvd = list(df[df['id'].isin(df.id[df.group == 'DVD'])]['id'])
f = open('Amazon0601.txt','r',encoding='utf-8')
lines = f.readlines()
lines = lines[4:] # élimination de l'en-tête
dvd = open('dvd.txt','w') # Ecriture des lignes qui nous intéressent dans un nouveau fichier dvd.txt
e = ThreadPoolExecutor()
e.map(is_dvd,lines)
dvd.close()
f.close()