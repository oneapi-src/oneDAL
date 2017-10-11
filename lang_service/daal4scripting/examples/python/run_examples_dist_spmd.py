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

# This files shows the use of SPMD-mode in daal4py.
# All processes execute the same program, so some commands need
# to be protected from execution on some processes.

import daal4py as d
from numpy import loadtxt
from time import gmtime, strftime

# calling init with spmd=True configures for SPMD processing
d.daalinit(spmd=True)

################################################################################################
############## kmeans
def kmeans():
    myfile = "../../samples/cpp/mpi/data/distributed/kmeans_dense.csv"
    mytable = loadtxt(myfile, delimiter=',')

    # Let's create the init algorithm only and delay execution
    centroids = d.kmeans_init(20, t_method="plusPlusDense", distributed=True)

    # we could use our centroids computed otherwise for kmeans
    return d.kmeans(20, distributed=True).compute(mytable, centroids.compute(mytable))

################################################################################################
############## PCA
def pca():
    files=["../../samples/cpp/mpi/data/distributed/pca_normalized_1.csv",
           "../../samples/cpp/mpi/data/distributed/pca_normalized_2.csv",
           "../../samples/cpp/mpi/data/distributed/pca_normalized_3.csv",
           "../../samples/cpp/mpi/data/distributed/pca_normalized_4.csv"]
    myfile = files[d.my_procid()]

    # Here we pass in a numpy-array
    return d.pca(distributed=True).compute(myfile)

################################################################################################
############## Naive Bayes
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
    mydfile  = dfiles[d.my_procid()]
    mylfile = lfiles[d.my_procid()]

    # Here simply pass in the "local" file-name(s)
    res = d.multinomial_naive_bayes_training(20, distributed=True).compute(mydfile, mylfile)

    if d.my_procid() == 0:
        return d.multinomial_naive_bayes_prediction(20).compute(loadtxt(tdfile, delimiter=','), res['model'])
    return None

################################################################################################
#### run all
for func, name in [(kmeans, 'kmeans'), (pca, 'pca'), (naive_bayes, 'naive bayes')]:
    try:
        if d.my_procid() == 0:
            print('####################### ' + name)
        res = func()
        if d.my_procid() == 0:
            assert '__daalptr__' in res
            print(strftime("%H:%M:%S", gmtime()) + '\tPASSED')
    except BaseException as ex:
        print(str(ex))
        print(strftime("%H:%M:%S", gmtime()) + '\tFAILED')

################################################################################################
# we need a fini to prevent processes to terminate prematurely
d.daalfini()
