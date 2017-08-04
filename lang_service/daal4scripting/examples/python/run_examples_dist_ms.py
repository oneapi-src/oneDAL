#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

# This files shows the use of master-slave-mode in daal4py.
# Only the root process executes this file. Distributed
# computation is initiated from the the root process.
# All other processes just sit there and wait for work (in daalinit).

import daal4py as d
from numpy import loadtxt
from time import gmtime, strftime

# calling init without arguments (-> spmd=False) configures for master-slave processing
d.daalinit()

################################################################################################
############## kmeans
# Here we pass in all partitions as tables

def kmeans():
    files = ["../../samples/cpp/mpi/data/distributed/kmeans_dense.csv"] * 4
    tables = [loadtxt(f, delimiter=',') for f in files]

    # let's use the plusplus method
    centroids = d.kmeans_init(20, t_method="plusPlusDense", distributed=True)

    # we could use our centroids computed otherwise for kmeans
    return d.kmeans(20, distributed=True).compute(tables, centroids.compute(tables))

################################################################################################
############## PCA
# Here we pass in all partitions as numpy-arrays

def pca():
    files=["../../samples/cpp/mpi/data/distributed/pca_normalized_1.csv",
           "../../samples/cpp/mpi/data/distributed/pca_normalized_2.csv",
           "../../samples/cpp/mpi/data/distributed/pca_normalized_3.csv",
           "../../samples/cpp/mpi/data/distributed/pca_normalized_4.csv"]

    return d.pca(distributed=True).compute([loadtxt(f, delimiter=',') for f in files])

################################################################################################
############## Naive Bayes
# Here we simply pass in the file-names, but we have several input files!

def naive_bayes():
    dfiles = ["../../samples/cpp/mpi/data/distributed/naivebayes_train_dense.csv",
              "../../samples/cpp/mpi/data/distributed/naivebayes_train_dense.csv",
              "../../samples/cpp/mpi/data/distributed/naivebayes_train_dense.csv",
              "../../samples/cpp/mpi/data/distributed/naivebayes_train_dense.csv"]
    lfiles = ["../../samples/cpp/mpi/data/distributed/naivebayes_train_labels.csv",
              "../../samples/cpp/mpi/data/distributed/naivebayes_train_labels.csv",
              "../../samples/cpp/mpi/data/distributed/naivebayes_train_labels.csv",
              "../../samples/cpp/mpi/data/distributed/naivebayes_train_labels.csv"]
    tdfile = "../../samples/cpp/mpi/data/distributed/naivebayes_test_dense.csv"
    tlfile = "../../samples/cpp/mpi/data/distributed/naivebayes_test_labels.csv"

    res = d.multinomial_naive_bayes_training(20, distributed=True).compute(dfiles, lfiles)

    # prediction does not distributed the computation
    return d.multinomial_naive_bayes_prediction(20).compute(loadtxt(tdfile, delimiter=','), res['model'])

################################################################################################
#### run all
for func, name in [(kmeans, 'kmeans'), (pca, 'pca'), (naive_bayes, 'naive bayes')]:
    try:
        print('####################### ' + name)
        assert '__daalptr__' in func()
        print(strftime("%H:%M:%S", gmtime()) + '\tPASSED')
    except BaseException as ex:
        print(str(ex))
        print(strftime("%H:%M:%S", gmtime()) + '\tFAILED')

################################################################################################
# we need a fini to prevent processes to terminate prematurely
d.daalfini()
