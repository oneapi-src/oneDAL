/* file: neural_networks_prediction_feedforward_impl.i */
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
using namespace daal::services;

template<typename algorithmFPType, Method method, CpuType cpu>
Status NeuralNetworksFeedforwardPredictionKernel<algorithmFPType, method, cpu>::initialize(
    const Input *input, const neural_networks::prediction::Parameter *parameter, Result *result)
{
    nLayers   = input->get(prediction::model)->getLayers()->size();
    nSamples  = input->get(prediction::data)->getDimensions().get(0);
    batchSize = parameter->batchSize;
    if (nSamples < batchSize) { return Status(); }

    /* Get the number of last layers in the network and their indeces */
    Collection<layers::NextLayers> *nextLayers = input->get(prediction::model)->getNextLayers().get();
    KeyValueDataCollectionPtr predictionCollectionPtr = result->get(prediction::predictionCollection);

    lastLayersIndices.reset(new LastLayerIndices(nextLayers, predictionCollectionPtr));
    DAAL_CHECK_MALLOC(lastLayersIndices.get() && lastLayersIndices->isValid())

    nLastLayers = lastLayersIndices->nLast(); /* number of last layers in the network */

    /* Create a tensor to pass as an input to the first forward layer in neural network */
    Collection<size_t> sampleSize = input->get(prediction::data)->getDimensions();
    sampleSize[0] = batchSize;
    Status s;
    sample = HomogenTensor<algorithmFPType>::create(sampleSize, Tensor::doNotAllocate, &s);
    DAAL_CHECK_STATUS_VAR(s);

    /* Initialize buffers to manage reading memory operations for the last layer results */
    lastLayerResults.reset(nLastLayers);
    DAAL_CHECK_MALLOC(lastLayerResults.get())

    /* Initialize buffers to manage writing memory operations for the prediction results */
    predictions.reset(nLastLayers);
    DAAL_CHECK_MALLOC(predictions.get())

    return Status();
}

/**
 *  \brief Kernel for Neural Network prediction
 */
template<typename algorithmFPType, Method method, CpuType cpu>
Status NeuralNetworksFeedforwardPredictionKernel<algorithmFPType, method, cpu>::compute(const Input *input, Result *result)
{
    Status s;
    ForwardLayersPtr forwardLayers = input->get(prediction::model)->getLayers();
    TensorPtr data = input->get(prediction::data);
    if (nSamples < batchSize) { return s; }

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
        const algorithmFPType *sampleArray = sampleSubtensor.next(0, 0, i, batchSize);
        DAAL_CHECK_BLOCK_STATUS(sampleSubtensor)
        sample->setArray(const_cast<algorithmFPType *>(sampleArray));

        /* Forward pass through the neural network */
        for(size_t layerId = 0; layerId < nLayers; layerId++)
        {
            layers::forward::LayerIfacePtr forwardLayer = forwardLayers->get(layerId);
            DAAL_CHECK_STATUS(s, processLayerErrors(layerId, forwardLayer->computeNoThrow()))
        }

        /* Copy results from the last layers into the user provided memory */
        for (size_t j = 0; j < nLastLayers; j++)
        {
            const algorithmFPType *lastLayerResultArray = lastLayerResults[j].next(0, 0, 0, batchSize);
            DAAL_CHECK_BLOCK_STATUS(lastLayerResults[j])
            algorithmFPType *predictionArray = predictions[j].next(0, 0, i, batchSize);
            DAAL_CHECK_BLOCK_STATUS(predictions[j])

            size_t blockSize = lastLayerResults[j].getSize() * sizeof(algorithmFPType);
            daal_memcpy_s(predictionArray, blockSize, lastLayerResultArray, blockSize);
        }
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status NeuralNetworksFeedforwardPredictionKernel<algorithmFPType, method, cpu>::reset()
{
    lastLayersIndices.reset();
    lastLayerResults.reset(0);
    predictions.reset(0);
    sample.reset();
    return Status();
}

} // namespace internal
} // namespace feedforward
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
