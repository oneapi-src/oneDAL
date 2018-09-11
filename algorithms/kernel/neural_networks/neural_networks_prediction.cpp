/* file: neural_networks_prediction.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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

//++
//  Implementation of neural networks calculation functions.
//--

#include "neural_networks_prediction_input.h"
#include "neural_networks_prediction_result.h"
#include "neural_networks_prediction_model.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace prediction
{
namespace interface1
{
__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_NEURAL_NETWORKS_PREDICTION_RESULT_ID);
__DAAL_REGISTER_SERIALIZATION_CLASS(Model, SERIALIZATION_NEURAL_NETWORKS_PREDICTION_MODEL_ID);
}

Input::Input() : daal::algorithms::Input(lastModelInputId + 1) {};
Input::Input(const Input& other) : daal::algorithms::Input(other){}

TensorPtr Input::get(TensorInputId id) const
{
    return Tensor::cast(Argument::get(id));
}

void Input::set(TensorInputId id, const TensorPtr &value)
{
    Argument::set(id, value);
}

ModelPtr Input::get(ModelInputId id) const
{
    return Model::cast(Argument::get(id));
}

void Input::set(ModelInputId id, const ModelPtr &value)
{
    Argument::set(id, value);
}

Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    const Parameter *param = static_cast<const Parameter *>(par);
    TensorPtr dataTensor = get(data);
    Status s;
    DAAL_CHECK_STATUS(s, checkTensor(dataTensor.get(), dataStr()));
    size_t nSamples = dataTensor->getDimensionSize(0);
    DAAL_CHECK_EX(nSamples >= param->batchSize, ErrorIncorrectParameter, ParameterName, batchSizeStr());
    DAAL_CHECK(get(model), ErrorNullModel);
    return s;
}


Result::Result() : daal::algorithms::Result(lastResultCollectionId + 1)
{
    set(predictionCollection, KeyValueDataCollectionPtr(new KeyValueDataCollection()));
}

TensorPtr Result::get(ResultId id) const
{
    KeyValueDataCollectionPtr collection = get(predictionCollection);
    if (!collection) { return TensorPtr(); }
    if (collection->size() == 0) { return TensorPtr(); }
    return Tensor::cast(collection->getValueByIndex(0));
}

KeyValueDataCollectionPtr Result::get(ResultCollectionId id) const
{
    return KeyValueDataCollection::cast(Argument::get(id));
}

TensorPtr Result::get(ResultCollectionId id, size_t key) const
{
    KeyValueDataCollectionPtr collection = get(id);
    if (!collection) { return TensorPtr(); }
    return Tensor::cast((*collection)[key]);
}

void Result::set(ResultId id, const TensorPtr &value)
{
    add(predictionCollection, 0, value);
}

void Result::set(ResultCollectionId id, const KeyValueDataCollectionPtr &value)
{
    Argument::set(id, value);
}

void Result::add(ResultCollectionId id, size_t key, const TensorPtr &value)
{
    KeyValueDataCollectionPtr collection = get(id);
    if (!collection) { return; }
    (*collection)[key] = value;
}

Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    KeyValueDataCollectionPtr predictionTensorCollection = get(predictionCollection);
    DAAL_CHECK(predictionTensorCollection, ErrorNullOutputDataCollection)
    DAAL_CHECK(predictionTensorCollection->size(), ErrorIncorrectNumberOfElementsInResultCollection)

    const prediction::Input *algInput = static_cast<const prediction::Input *>(input);
    TensorPtr dataTensor = algInput->get(prediction::data);

    size_t nSamples = dataTensor->getDimensionSize(0);

    size_t nLastLayers = predictionTensorCollection->size();
    for (size_t i = 0; i < nLastLayers; i++)
    {
        size_t layerId = predictionTensorCollection->getKeyByIndex((int)i);
        TensorPtr predictionTensor = get(predictionCollection, layerId);
        DAAL_CHECK_EX(predictionTensor, ErrorNullTensor, ArgumentName, predictionStr())

        Collection<size_t> expectedDims = predictionTensor->getDimensions();
        expectedDims[0] = nSamples;
        Status s;
        DAAL_CHECK_STATUS(s, checkTensor(predictionTensor.get(), predictionStr(), &expectedDims));
    }

    return Status();
}

}
}
}
}
