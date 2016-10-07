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

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace prediction
{
namespace internal
{
template<typename algorithmFPType, Method method, CpuType cpu>
void NeuralNetworksFeedforwardPredictionKernel<algorithmFPType, method, cpu>::initialize(
    const Input *input, const neural_networks::prediction::Parameter *parameter, Result *result)
{
    nLayers   = input->get(prediction::model)->getLayers()->size();
    nSamples  = input->get(prediction::data)->getDimensions().get(0);
    batchSize = parameter->batchSize;
    if (nSamples < batchSize) { return; }

    /* Get the number of last layers in the network and their indeces */
    Collection<layers::NextLayers> *nextLayers = input->get(prediction::model)->getNextLayers().get();
    KeyValueDataCollectionPtr predictionCollectionPtr = result->get(prediction::predictionCollection);

    lastLayersIndices = new LastLayerIndices(nextLayers, predictionCollectionPtr);
    if (lastLayersIndices->getError())
    {
        reset();
        this->_errors->add(ErrorMemoryAllocationFailed); return;
    }

    nLastLayers = lastLayersIndices->nLast(); /* number of last layers in the network */

    /* Create a tensor to pass as an input to the first forward layer in neural network */
    Collection<size_t> sampleSize = input->get(prediction::data)->getDimensions();
    sampleSize[0] = batchSize;
    sample.reset(new HomogenTensor<algorithmFPType>(sampleSize, Tensor::notAllocate));

    /* Initialize buffers to manage reading memory operations for the last layer results */
    lastLayerResults = new ReadSubtensor<algorithmFPType, cpu>[nLastLayers];

    /* Initialize buffers to manage writing memory operations for the prediction results */
    predictions = new WriteOnlySubtensor<algorithmFPType, cpu>[nLastLayers];
    if (!lastLayerResults || !predictions)
    {
        reset();
        this->_errors->add(ErrorMemoryAllocationFailed); return;
    }
}

/**
 *  \brief Kernel for Neural Network prediction
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void NeuralNetworksFeedforwardPredictionKernel<algorithmFPType, method, cpu>::compute(const Input *input, Result *result)
{
    SharedPtr<ForwardLayers> forwardLayers = input->get(prediction::model)->getLayers();
    TensorPtr data = input->get(prediction::data);
    if (nSamples < batchSize) { return; }

    forwardLayers->get(0)->getLayerInput()->set(forward::data, sample);

    /* Buffer that manages reading memory operations for the input data tensor */
    ReadSubtensor<algorithmFPType, cpu> sampleSubtensor(data.get(), 0, 0, 0, 0);

    for (size_t i = 0; i < nLastLayers; i++)
    {
        TensorPtr valueTensor = forwardLayers->get(lastLayersIndices->layerIndex(i))->getLayerResult()->get(forward::value);
        lastLayerResults[i].set(*valueTensor, 0, 0, 0, 0);
    }

    /* Initialize buffers to manage writing memory operations for the prediction results */
    for (size_t i = 0; i < nLastLayers; i++)
    {
        TensorPtr predictionTensor = result->get(prediction::predictionCollection, lastLayersIndices->tensorIndex(i));
        predictions[i].set(*predictionTensor, 0, 0, 0, 0);
    }

    for(size_t i = 0; i < nSamples - batchSize + 1; i += batchSize)
    {
        /* Retrieve next batch of input data and pass it to the first layer */
        sample->setArray(const_cast<algorithmFPType *>(sampleSubtensor.next(0, 0, i, batchSize)));

        /* Forward pass through the neural network */
        for(size_t layerId = 0; layerId < nLayers; layerId++)
        {
            layers::forward::LayerIfacePtr forwardLayer = forwardLayers->get(layerId);
            forwardLayer->computeNoThrow();
            if (!processLayerErrors(layerId, forwardLayer->getErrors()->getErrors(), this->_errors))
            {
                reset();
                return;
            }
        }

        /* Copy results from the last layers into the user provided memory */
        for (size_t j = 0; j < nLastLayers; j++)
        {
            const algorithmFPType *lastLayerResultArray = lastLayerResults[j].next(0, 0, 0, batchSize);
            algorithmFPType *predictionArray = predictions[j].next(0, 0, i, batchSize);

            size_t blockSize = lastLayerResults[j].getSize() * sizeof(algorithmFPType);
            services::daal_memcpy_s(predictionArray, blockSize, lastLayerResultArray, blockSize);
        }
    }
}

template<typename algorithmFPType, Method method, CpuType cpu>
void NeuralNetworksFeedforwardPredictionKernel<algorithmFPType, method, cpu>::reset()
{
    if(lastLayersIndices) { delete lastLayersIndices; lastLayersIndices = NULL; }
    if(lastLayerResults)  { delete [] lastLayerResults; lastLayerResults = NULL; }
    if(predictions)       { delete [] predictions; predictions = NULL; }
    sample.reset();
}

} // namespace internal
} // namespace feedforward
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
