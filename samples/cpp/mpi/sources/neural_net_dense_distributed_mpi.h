/* file: neural_net_dense_distributed_mpi.h */
/*******************************************************************************
* Copyright 2017-2018 Intel Corporation.
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
*
* License:
* http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
* eement/
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
    size_t fc3;
    size_t sm;
};

training::TopologyPtr configureNet(LayerIds* ids = NULL)
{
    /* Create layers of the neural network */
    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer1(new fullyconnected::Batch<>(20));

    fullyConnectedLayer1->parameter.weightsInitializer.reset(new initializers::uniform::Batch<>(-0.001, 0.001));

    fullyConnectedLayer1->parameter.biasesInitializer.reset(new initializers::uniform::Batch<>(0, 0.5));

    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer2(new fullyconnected::Batch<>(40));

    fullyConnectedLayer2->parameter.weightsInitializer.reset(new initializers::uniform::Batch<>(0.5, 1));

    fullyConnectedLayer2->parameter.biasesInitializer.reset(new initializers::uniform::Batch<>(0.5, 1));

    /* Create fully-connected layer and initialize layer parameters */
    SharedPtr<fullyconnected::Batch<> > fullyConnectedLayer3(new fullyconnected::Batch<>(2));

    fullyConnectedLayer3->parameter.weightsInitializer.reset(new initializers::uniform::Batch<>(-0.005, 0.005));

    fullyConnectedLayer3->parameter.biasesInitializer.reset(new initializers::uniform::Batch<>(0, 1));

    /* Create softmax layer and initialize layer parameters */
    SharedPtr<loss::softmax_cross::Batch<> > softmaxCrossEntropyLayer(new loss::softmax_cross::Batch<>());

    /* Create topology of the neural network */
    training::TopologyPtr topology(new training::Topology());

    /* Add layers to the topology of the neural network */
    const size_t fc1 = topology->add(fullyConnectedLayer1);
    const size_t fc2 = topology->add(fullyConnectedLayer2);
    const size_t fc3 = topology->add(fullyConnectedLayer3);
    const size_t sm = topology->add(softmaxCrossEntropyLayer);
    topology->get(fc1).addNext(fc2);
    topology->get(fc2).addNext(fc3);
    topology->get(fc3).addNext(sm);
    if(ids)
    {
        ids->fc1 = fc1;
        ids->fc2 = fc2;
        ids->fc3 = fc3;
        ids->sm = sm;
    }
    return topology;
}
