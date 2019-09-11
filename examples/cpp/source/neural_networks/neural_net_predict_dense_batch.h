/* file: neural_net_predict_dense_batch.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::services;

/** Structure that contains neural network layers identifiers */
struct LayerIds
{
    size_t fc1;
    size_t fc2;
    size_t sm;
};

prediction::TopologyPtr configureNet(LayerIds* ids = NULL)
{
    /* Create layers of the neural network */
    /* Create first fully-connected layer */
    SharedPtr<fullyconnected::forward::Batch<> > fullyConnectedLayer1(new fullyconnected::forward::Batch<>(5));

    /* Create second fully-connected layer */
    SharedPtr<fullyconnected::forward::Batch<> > fullyConnectedLayer2(new fullyconnected::forward::Batch<>(2));

    /* Create softmax layer */
    SharedPtr<softmax::forward::Batch<> > softmaxLayer(new softmax::forward::Batch<>());

    /* Create topology of the neural network */
    prediction::TopologyPtr topology(new prediction::Topology());

    /* Add layers to the topology of the neural network */
    const size_t fc1 = topology->add(fullyConnectedLayer1);
    const size_t fc2 = topology->add(fullyConnectedLayer2);
    const size_t sm = topology->add(softmaxLayer);
    topology->get(fc1).addNext(fc2);
    topology->get(fc2).addNext(sm);
    if(ids)
    {
        ids->fc1 = fc1;
        ids->fc2 = fc2;
        ids->sm = sm;
    }
    return topology;
}
