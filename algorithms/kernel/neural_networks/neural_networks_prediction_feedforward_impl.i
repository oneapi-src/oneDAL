/* file: neural_networks_prediction_feedforward_impl.i */
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

/*
//++
//  Implementation of feedforward algorithm
//--
*/

#ifndef __NEURAL_NETWORKS_PREDICTION_FEEDFORWARD_IMPL_I__
#define __NEURAL_NETWORKS_PREDICTION_FEEDFORWARD_IMPL_I__

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
using namespace layers;
namespace prediction
{
namespace internal
{

/**
 *  \brief Kernel for Neural Network prediction
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void NeuralNetworksFeedforwardPredictionKernel<algorithmFPType, method, cpu>::compute(
    const Input *input, const neural_networks::prediction::Parameter *parameter, Result *result)
{
    SharedPtr<Model> model = input->get(prediction::model);
    SharedPtr<ForwardLayers> forwardLayers = model->getLayers();
    size_t nLayers = forwardLayers->size();

    SharedPtr<Tensor> data = input->get(prediction::data);
    SharedPtr<Tensor> predictionResults = result->get(prediction::prediction);
    size_t nSamples = data->getDimensions().get(0);

    SharedPtr<forward::Result> lastLayerResult = forwardLayers->get(nLayers - 1)->getLayerResult();
    SubtensorDescriptor<algorithmFPType> valueSubtensor, predictionSubtensor;

    Collection<size_t> sampleSize = data->getDimensions();
    sampleSize[0] = 1;

    SharedPtr<HomogenTensor<algorithmFPType> > sample(new HomogenTensor<algorithmFPType>(sampleSize, Tensor::notAllocate));
    SubtensorDescriptor<algorithmFPType> sampleSubtensor;

    forwardLayers->get(0)->getLayerInput()->set(forward::data, sample);

    for(size_t i = 0; i < nSamples; i++)
    {
        data->getSubtensor(0, 0, i, 1, readOnly, sampleSubtensor);
        sample->setArray(sampleSubtensor.getPtr());

        for(size_t layerId = 0; layerId < nLayers; layerId++)
        {
            forwardLayers->get(layerId)->computeNoThrow();
            if(forwardLayers->get(layerId)->getErrors()->size() != 0) {this->_errors->add(forwardLayers->get(layerId)->getErrors()->getErrors()); return;}
        }
        SharedPtr<Tensor> valueTensor = lastLayerResult->get(forward::value);
        valueTensor->getSubtensor(0, 0, 0, 1, readOnly, valueSubtensor);
        predictionResults->getSubtensor(0, 0, i, 1, writeOnly, predictionSubtensor);
        algorithmFPType *valueArray = valueSubtensor.getPtr();
        algorithmFPType *resultArray = predictionSubtensor.getPtr();

        size_t valueSize = valueSubtensor.getSize();
        services::daal_memcpy_s(resultArray, valueSize * sizeof(algorithmFPType), valueArray, valueSize * sizeof(algorithmFPType));

        predictionResults->releaseSubtensor(predictionSubtensor);
        valueTensor->releaseSubtensor(valueSubtensor);
        data->releaseSubtensor(sampleSubtensor);
    }
}

} // namespace internal
} // namespace feedforward
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
