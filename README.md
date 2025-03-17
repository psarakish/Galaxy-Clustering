# Galaxy Clustering in 3D space

This project involve a clustering analysis, on the [HECATE](https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1896K/abstract) galaxy catalog 
based on the galaxies spatial arrangement, with the intention of assigning cluster identifiers to each galaxy candidate. The project builds upon the
Master's Thesis titled ["Galaxy cluster detection in the local universe, using machine learning methods"](https://elocus.lib.uoc.gr//dlib/e/1/5/metadata-dlib-1698042506-220193-30897.tkl).
The proposed methods include the use of DBSCAN, Agglomerative Clustering and a combination (stacking) of these two clustering algorithms. 

## Data

[HECATE (The Heraklion Extragalactic CATaloguE) (K. Kovlakas, A. Zezas, J. Andrews
et al. 2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.1896K/abstract) is an all-sky value-added galaxy catalog of
204,733 individual galaxies within a radius of 200 Mpc (Redshift/z=0.05). HECATE is based on the HyperLEDA catalog and is enriched with additional information from other extragalactic and photometric catalogues. The catalog aims to
support contemporary and upcoming multi-wavelength investigations of the nearby universe. It offers lots of information such as positions, distances, sizes, photometric measurements etc., but not cluster associations.

In order to develop and test the methods presented in this project, local universe galactic catalogs of known clusters were also used, like:

* [Northern/Southern Abell catalogue (G. Abell, H. Corwin et al. 1989)](https://ui.adsabs.harvard.edu/abs/1989ApJS...70....1A/abstract)
* [Compact Groups of Galaxies (A. McConnachie, D. Patton et al. 2009)](https://ui.adsabs.harvard.edu/abs/2009MNRAS.395..255M/abstract)
* [Hickson Compact Groups of Galaxies (P. Hickson 1982)](https://ui.adsabs.harvard.edu/abs/1982ApJ...255..382H/abstract)
* [Extended Virgo Cluster Catalog (K. Suk, R. Soo-Chang et al. 2014)](https://ui.adsabs.harvard.edu/abs/2014ApJS..215...22K/abstract)
* Abell catalog, a complete search was performed on [NED (NASA/IPAC Extragalactic Database)](https://ned.ipac.caltech.edu/) with ’ABELL’ cross-ID

## Methodology

Clustering the entire HECATE catalog is a computationally expensive process. Moreover, determining the optimal hyperparameters for the clustering algorithms is not straightforward. The availability of cluster labels for a small subset of HECATE presented an opportunity to apply **semi-supervised** methods. These are nearby galaxies that are known to reside big clusters, such as Coma, Hydra and Virgo. In this approach, that domain knowledge, specifically the known cluster labels for certain galaxies, was leveraged. This allowed to effectively assess how the clustering algorithms identified these cluster structures, potential galaxy alignments, and projection effects.

### *Selection of galaxies*

To find the galaxy subset with the known cluster labels (`Clusteringdata.csv`), a series of data selection techniques were applied on the catalogs mentioned above.

* Through [TOPCAT (Tool for OPerations on Catalogues And Tables)](https://www.star.bris.ac.uk/~mbt/topcat/) all HECATE galaxies were cross matched with the galaxies of the other catalogs.

* The resulted galaxy subset was refined by applying velocity based filtering criteria.

A total of **11517** HECATE galaxies were found to be part of well known nearby clusters or groups and were involved in the **semi-supervised** method proposed.

###  *Distance Metric* 

Each galaxy was characterized by four features: `RA`, `DEC`, `Redshift` and `cluster_label`. To determine the pairwise distances for every galaxy pair, a custom
distance function was created, adapted for spherical coordinates (`Distance_metric.py`):

$$
D = \sqrt{r^2 + r'^2 - 2r r' (\sin\theta \sin\theta' \cos(\phi - \phi') + \cos\theta \cos\theta')}
$$

where `(r,θ,φ)` and `(r',θ',φ')` spherical coordinates of a galaxy pair. The distance r and r' was calculated
using Hubble's Law for small redshift values,

$$
r = \frac{u_{rs}}{H_0}
$$

where $H_{0}=67 km/s/Mpc$ is the Hubble's constant, and

$$
u_{rs} = zc
$$

>[!NOTE]
>Assuming the expansion of the local universe is linear and distances satisfy the triangular inequality.

### _Evaluation_

Given the luxury provided by the ground truth labeled galaxies, a suitable assessment metric would be the creation of a custom scoring function based on F1-score optimization which would evaluate the agreement between ground truth and predicted galaxy cluster labels (`Evaluation.py`).

Here is an overview of the function’s implementation:
* For a true cluster in the labeled set, the function finds the galaxies residing in it
* It extracts the predicted cluster labels of these galaxies
* Computes the true positives, false positives and false negatives by assuming two classes, one for the specific predicted cluster being investigated and all the others as a second class
* Computes the F1score for each predicted cluster and stores the predicted cluster with the highest value
* When the scan is completed, the function continue to the next true cluster following the same procedure
* For all true clusters in the labeled set, it calculates the mean between all maximum F1<sub>score<sub>

### _Clustering Method_

Assuming that the minimun number of galaxies for a cluster/group to be identified is ``4``, there are ``190 different clusters/groups`` on the labeled subset. The clustering algorithms were assessed on how well they predicted the number of ground truth clusters while also achieving a good performance (`Evaluation.py`).

The labeled dataset was split into ``train`` and ``test`` samples, ``80%-20%`` respectively. The same galaxies belonging to train set for ``DBSCAN`` ,
was the same for ``AgglomerativeClustering``. In that way a fair and accurate comparison is ensured among the algorithms evaluation performance and clustering results. When DBSCAN and Agglomerative clustering were combined on a ``stacking approach``, they had to be trained on different training sets in a way to avoid ``overfitting``. So the dataset was first divided into ``80%-20%`` train and test samples, and then the train set was further split into ``40%-40%``. Each half was used to train each clustering algorithm. On all cases the best models were evaluated on
the 20% test sample.

>[!IMPORTANT]
> Upon dividing the initial dataset into an 80%-20% split, a notable issue emerged. Consider a scenario involving a group of four galaxies. The splitting process could potentially >allocate one galaxy to the test sample, leaving the remaining three for training, or the other way around. Under such circumstances, given the present assumption that a cluster needs 4 >galaxies to be identified, DBSCAN is likely to classify these isolated galaxies as outliers, leading to the loss of that particular group. So to avoid missing any small clusters of >galaxies, the best model acquired by both clustering algorithms was ``re-fit`` on the complete set of galaxies. In that way all galaxies were assigned their predicted match. However, >>this >approach inherently risked ``overfitting``, because the proper approach would require the model to be applied on a separate test set. For the purposes of this project, the primary
>focus was to examine the behavior of the labeled data and to create a model that best describes them. In this specific context, the potential overfitting is not of great concern.

#### [*DBSCAN*](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together data points that are in close proximity based on a distance measure (``eps``) and a specified density, i.e minimum number of points (``min samples``) 

* Pros
* Cons

#### [*Agglomerative Clustering*](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)

**Agglomerative clustering** is the most common algorithm of *Hierachical clustering*, which adopts a ”bottom-up” strategy. Starting with each data point as a single cluster, it 
recursively merges the closest pair of clusters into a single cluster, continuing this process until only one large cluster remains or a stopping criterion is met (``distance 
threshold``).

* Pros
* Cons

#### *Stacking Approach*

* Pros
* Cons
