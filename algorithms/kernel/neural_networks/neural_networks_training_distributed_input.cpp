/* file: neural_networks_training_distributed_input.cpp */
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

#include "neural_networks_training_input.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
DistributedInput<step1Local>::DistributedInput(size_t nElements) : Input(nElements) {};
DistributedInput<step1Local>::DistributedInput(const DistributedInput& other) : Input(other){}

ModelPtr DistributedInput<step1Local>::get(Step1LocalInputId id) const
{
    return Model::cast(Argument::get(id));
}

void DistributedInput<step1Local>::set(Step1LocalInputId id, const ModelPtr &value)
{
    Argument::set(id, value);
}

Status DistributedInput<step1Local>::check(const daal::algorithms::Parameter *par, int method) const
{
    ModelPtr model = get(inputModel);
    TensorPtr dataTensor = get(data);
    Status s;
    DAAL_CHECK_STATUS(s, checkTensor(dataTensor.get(), dataStr()))

    size_t nSamples = dataTensor->getDimensionSize(0);
    size_t modelBatchSize = model->getForwardLayer(0)->getLayerInput()->get(layers::forward::data)->getDimensionSize(0);
    DAAL_CHECK_EX(nSamples >= modelBatchSize, ErrorIncorrectParameter, ParameterName, batchSizeStr());

    DAAL_CHECK_STATUS(s, checkImpl(par, method));
    return s;
}


DistributedInput<step2Master>::DistributedInput() : daal::algorithms::Input(lastStep2MasterInputId + 1)
{
    set(partialResults, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

DistributedInput<step2Master>::DistributedInput(const DistributedInput& other) : daal::algorithms::Input(other){}

KeyValueDataCollectionPtr DistributedInput<step2Master>::get(Step2MasterInputId id) const
{
    return KeyValueDataCollection::cast(Argument::get(id));
}

void DistributedInput<step2Master>::set(Step2MasterInputId id, const KeyValueDataCollectionPtr &value)
{
    Argument::set(id, value);
}

void DistributedInput<step2Master>::add(Step2MasterInputId id, size_t key, const PartialResultPtr &value)
{
    KeyValueDataCollectionPtr collection = get(id);
    if (!collection) { return; }
    (*collection)[key] = value;
}

Status DistributedInput<step2Master>::check(const daal::algorithms::Parameter *par, int method) const
{
    return Status();
}

}
}
}
}
