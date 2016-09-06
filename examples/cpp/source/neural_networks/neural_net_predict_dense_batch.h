/* file: neural_net_predict_dense_batch.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::services;

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
