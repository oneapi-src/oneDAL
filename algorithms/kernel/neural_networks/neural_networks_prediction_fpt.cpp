/* file: neural_networks_prediction_fpt.cpp */
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

#include "neural_networks_prediction_result.h"
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
template<typename algorithmFPType>
DAAL_EXPORT Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Input *in = static_cast<const Input * >(input);

    ModelPtr predictionModel = in->get(model);
    Parameter *par = static_cast<Parameter *>(const_cast<daal::algorithms::Parameter *>(parameter));
    Collection<size_t> sampleSize = in->get(data)->getDimensions();
    sampleSize[0] = par->batchSize;

    predictionModel->allocate<algorithmFPType>(sampleSize, parameter);

    ForwardLayersPtr layers = predictionModel->getLayers();
    SharedPtr<Collection<layers::NextLayers> > nextLayers = predictionModel->getNextLayers();
    size_t nLayers = layers->size();
    Collection<size_t> lastLayerIds;
    for (size_t layerId = 0; layerId < nLayers; layerId++)
    {
        if (nextLayers->get(layerId).size() == 0)
        {
            lastLayerIds.push_back(layerId);
        }
    }

    size_t nLastLayers = lastLayerIds.size();
    size_t nResults = in->get(data)->getDimensionSize(0);
    Status s;
    for (size_t i = 0; i < nLastLayers; i++)
    {
        size_t layerId = lastLayerIds[i];
        layers::forward::ResultPtr lastLayerResult = layers->get(layerId)->getLayerResult();
        DAAL_CHECK_EX(lastLayerResult && lastLayerResult->get(layers::forward::value), ErrorNullTensor, ArgumentName, valueStr());
        Collection<size_t> resultDimensions = lastLayerResult->get(layers::forward::value)->getDimensions();
        resultDimensions[0] = in->get(data)->getDimensions().get(0);

        add(predictionCollection, layerId, HomogenTensor<algorithmFPType>::create(resultDimensions, Tensor::doAllocate, &s));
        DAAL_CHECK_STATUS_VAR(s);
    }
    return s;
}

template DAAL_EXPORT Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
}
}
}
}
