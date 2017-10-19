# Fuzzy clustering by Local Approximation of MEmbership

This is my own basic python implementation of the FLAME fuzzy clustering algorithm invented by Limin Fu and Enzo Medico and published in 
[Fu, L. and Medico, E., 2007. FLAME, a novel fuzzy clustering method for the analysis of DNA microarray data. *BMC bioinformatics*, 8(1), p.3. Vancouver](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-3). I was assisted in my understanding of the algorithm by [this helpful video](https://vimeo.com/78348227) by Max Schwenzer on Vimeo.

Included is [FLAME.py](FLAME.py) which implements the algorithm with a Scikit-learn style interface and a Jupyter notebook [iris.ipynb](iris.ipynb) which demonstrates the application of the algorithm to the iris dataset and to a make_blobs style dataset.

Dependencies are numpy, scipy and sklearn and Jupyter to run the notebook.