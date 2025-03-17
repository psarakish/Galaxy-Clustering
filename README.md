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

Given the luxury provided by the ground truth labeled galaxies, a suitable assessment metric would be the creation of a custom scoring function based on F1-score optimization which would evaluate the agreement between ground truth and predicted galaxy cluster labels.

### _Clustering Method_

Assuming that the minimun number of galaxies for a cluster/group to be identified is ``four``, there are 190 different clusters/groups on the labeled subset. The clustering algorithms were assessed on how well they predicted the number of ground truth clusters while also achieving a good performance.









## Results

add table of hyperparameters 


### *DBSCAN* add info/links on sklearn website etc etc
* Pros
* Cons
### *Agglomerative Clustering*
* Pros
* Cons
### *Stacking Approach*
* Pros
* Cons
