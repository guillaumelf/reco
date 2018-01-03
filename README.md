# reco
This is a small project showing how graph theory can help identify central products (in a marketing view) and make recommendations.

## What this project is about

The main topic of this project was to choose a concrete case where graph theory could be applied.
Here, we are dealing with **Amazon** products which are often bought together. The data used in this application results in a directed graph : when a product *i* is frequently bought with a product *j*, there is a directed link from *i* towards *j*.

For multiple reasons, the focus in this study will be on DVDs only. The main target is to identify the most "popular" products and make prediction on which articles could be linked with those aforementioned products.

To realise it, we will use **Python** and its library *NetworkX* which has been built in order to deal with graphs manipulation.

## The content of this repositery

You will find python scripts which helped perform this study by selecting and preprocessing the data we wished to work on, as well as calculating various centrality measures and link prediction for products. You can adapt this code to your particular problem.
There is also a report (in french) which explains graph theory, the main concepts and how the measures used in this study work. It also sums up the results we get by applying the chosen methods on our data.

**NOTE** : the *Amazon0601.txt* & *amazon-meta.txt* data were too voluminous and hence couldn't be uploaded on this repositery, but you can download them at http://snap.stanford.edu/data/amazon0601.html
