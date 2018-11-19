/* file: neural_networks_training_feedforward_impl.i */
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

#ifndef __NEURAL_NETWORKS_TRAINING_FEEDFORWARD_IMPL_I__
#define __NEURAL_NETWORKS_TRAINING_FEEDFORWARD_IMPL_I__

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{
namespace internal
{
using namespace daal::services;

/**
 *  \brief Kernel for Neural Network training in batch processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelBatch<algorithmFPType, method, cpu>::initialize(
    Tensor* data, Model* nnModel,
    const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
    const neural_networks::training::Parameter *parameter)
{
    Status s;
    DAAL_CHECK_STATUS(s, this->initializeBase(data, nnModel, groundTruthCollectionPtr, parameter))

    ForwardLayersPtr forwardLayers = nnModel->getForwardLayers();
    learnableLayerIndices.reset(new LearnableLayerIndices(forwardLayers.get()));
    DAAL_CHECK_MALLOC(learnableLayerIndices.get() && learnableLayerIndices->isValid())

    oneTableForAllWeights = nnModel->getWeightsAndBiasesStorageStatus();
    size_t nSolvers = (oneTableForAllWeights ? 1 : learnableLayerIndices->nLearnable());

    solvers.reset(nSolvers);
    DAAL_CHECK_MALLOC(solvers.get())

    for (size_t i = 0; i < nSolvers; i++)
    {
        DAAL_CHECK_STATUS(s, solvers[i].init(parameter->optimizationSolver))
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelBatch<algorithmFPType, method, cpu>::compute(
    Tensor* data, Model* nnModel,
    const KeyValueDataCollectionPtr &groundTruthCollectionPtr)
{
    Status s;
    const size_t nSolvers = solvers.size();
    for(size_t i = 0; i < nSolvers; i++)
    {
        DAAL_CHECK_STATUS(s, solvers[i].setSolverOptionalResult(nnModel->getSolverOptionalArgument(i)))
    }

    DAAL_CHECK_STATUS(s, this->computeBase(data, nnModel, groundTruthCollectionPtr))

    for(size_t i = 0; i < nSolvers; i++)
    {
        nnModel->setSolverOptionalArgument(solvers[i].getSolverOptionalResult(), i);
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelBatch<algorithmFPType, method, cpu>::updateWeights(Model& nnModel)
{
    Status s;
    if (oneTableForAllWeights)
    {
        Solver<algorithmFPType> &solver = solvers[0];
        DAAL_CHECK_STATUS(s, solver.updateWeightsAndBiases(
            nnModel.getWeightsAndBiases(), nnModel.getWeightsAndBiasesDerivatives()))

        nnModel.setWeightsAndBiases(solver.getMinimum());
    }
    else
    {
        for(size_t j = 0; j < solvers.size(); j++)
        {
            size_t layerId = learnableLayerIndices->layerIndex(j);
            Solver<algorithmFPType> &solver = solvers[j];
            DAAL_CHECK_STATUS(s, solver.updateWeightsAndBiases(
                nnModel.getWeightsAndBiases(layerId), nnModel.getWeightsAndBiasesDerivatives(layerId)))

            nnModel.setWeightsAndBiases(layerId, solver.getMinimum());
        }
    }
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
size_t TrainingKernelBatch<algorithmFPType, method, cpu>::getMaxIterations(size_t nSamples, size_t batchSizeParam) const
{
    size_t maxIterations = nSamples / batchSizeParam;
    size_t nIterationSolver = solvers[0].getNIterations();
    if(nIterationSolver && nIterationSolver < maxIterations)
    {
        maxIterations = nIterationSolver;
    }
    return maxIterations;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelBatch<algorithmFPType, method, cpu>::reset()
{
    Status s;
    DAAL_CHECK_STATUS(s, this->resetBase())
    solvers.reset(0);
    learnableLayerIndices.reset();
    return s;
}

/**
 *  \brief Kernel for Neural Network training in distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelDistributed<algorithmFPType, method, cpu>::initialize(
    Tensor* data, Model* nnModel,
    const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
    const neural_networks::training::Parameter *parameter)
{
    return this->initializeBase(data, nnModel, groundTruthCollectionPtr, parameter);
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelDistributed<algorithmFPType, method, cpu>::compute(
    Tensor* data, Model* nnModel,
    const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
    PartialResult *partialResult,
    const neural_networks::training::Parameter *parameter)
{
    Status s;
    DAAL_CHECK_STATUS(s, this->computeBase(data, nnModel, groundTruthCollectionPtr))

    partialResult->set(derivatives, nnModel->getWeightsAndBiasesDerivatives());

    WriteRows<algorithmFPType, cpu> batchSizeBlock(*(partialResult->get(batchSize)), 0, 1);
    DAAL_CHECK_BLOCK_STATUS(batchSizeBlock)
    algorithmFPType* batchSizeArray = batchSizeBlock.get();
    batchSizeArray[0] = nnModel->getForwardLayer(0)->getLayerInput()->get(layers::forward::data)->getDimensionSize(0);
    return s;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelDistributed<algorithmFPType, method, cpu>::updateWeights(Model& nnModel)
{
    return Status();
}

template<typename algorithmFPType, Method method, CpuType cpu>
size_t TrainingKernelDistributed<algorithmFPType, method, cpu>::getMaxIterations(size_t nSamples, size_t batchSizeParam) const
{
    return nSamples / batchSizeParam;
}

template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelDistributed<algorithmFPType, method, cpu>::reset()
{
    return this->resetBase();
}


/**
 *  \brief Kernel for Neural Network training in distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
Status TrainingKernelDistributedStep2<algorithmFPType, method, cpu>::compute(
    KeyValueDataCollection* collection,
    const neural_networks::training::Parameter *parameter,
    Model* nnModel)
{
    using namespace optimization_solver;

    size_t nPartialResults = collection->size();

    NumericTablePtr weightsAndBiasesDerivatives;
    if (nPartialResults == 1)
    {
        weightsAndBiasesDerivatives = PartialResult::cast(collection->getValueByIndex(0))->get(derivatives);
    }
    else
    {
        PartialResultPtr partialResults = PartialResult::cast((*collection)[0]);
        NumericTablePtr partialDerivative = partialResults->get(derivatives);
        NumericTablePtr batchSize = partialResults->get(training::batchSize);

        size_t derivSize = partialDerivative->getNumberOfRows();

        Status st;
        SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > fullDerivative = HomogenNumericTableCPU<algorithmFPType, cpu>::create(1, derivSize, &st);
        DAAL_CHECK_STATUS_VAR(st);

        weightsAndBiasesDerivatives = fullDerivative;
        algorithmFPType* derData = fullDerivative->getArray();
        algorithmFPType sum = 0;

        ReadRows<algorithmFPType, cpu> pDerRows(partialDerivative.get(), 0, derivSize);
        DAAL_CHECK_BLOCK_STATUS(pDerRows)
        const algorithmFPType* pDerData = pDerRows.get();

        ReadRows<algorithmFPType, cpu> batchSizeBlock(batchSize.get(), 0, 1);
        DAAL_CHECK_BLOCK_STATUS(batchSizeBlock)
        const algorithmFPType* batchSizeArray = batchSizeBlock.get();

        for (size_t j = 0; j < derivSize; j++)
        {
            derData[j] = batchSizeArray[0] * pDerData[j];
        }
        sum = batchSizeArray[0];

        for (size_t i = 1; i < nPartialResults; i++)
        {
            partialResults = PartialResult::cast((*collection)[i]);
            partialDerivative = partialResults->get(derivatives);
            batchSize = partialResults->get(training::batchSize);

            ReadRows<algorithmFPType, cpu> pDerRows(partialDerivative.get(), 0, derivSize);
            DAAL_CHECK_BLOCK_STATUS(pDerRows)
            const algorithmFPType* pDerData = pDerRows.get();

            ReadRows<algorithmFPType, cpu> batchSizeBlock(batchSize.get(), 0, 1);
            DAAL_CHECK_BLOCK_STATUS(batchSizeBlock)
            const algorithmFPType* batchSizeArray = batchSizeBlock.get();

            for (size_t j = 0; j < derivSize; j++)
            {
                derData[j] += batchSizeArray[0] * pDerData[j];
            }
            sum += batchSizeArray[0];
        }

        algorithmFPType invNPartialResults = 1.0 / sum;
        for (size_t j = 0; j < derivSize; j++)
        {
            derData[j] *= invNPartialResults;
        }
    }

    Solver<algorithmFPType> solver;
    Status s;
    DAAL_CHECK_STATUS(s, solver.init(parameter->optimizationSolver))
    DAAL_CHECK_STATUS(s, solver.setSolverOptionalResult(nnModel->getSolverOptionalArgument(0)))
    DAAL_CHECK_STATUS(s, solver.updateWeightsAndBiases(nnModel->getWeightsAndBiases(), weightsAndBiasesDerivatives))
    nnModel->setWeightsAndBiases(solver.getMinimum());
    nnModel->setSolverOptionalArgument(solver.getSolverOptionalResult(), 0);
    return s;
}

/**
 *  \brief Kernel for Neural Network training
 */
template<typename algorithmFPType, CpuType cpu>
Status TrainingKernelBase<algorithmFPType, cpu>::initializeBase(
    Tensor *data, Model *nnModel,
    const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
    const neural_networks::training::Parameter *parameter)
{
    ForwardLayersPtr forwardLayers = nnModel->getForwardLayers();
    batchSizeParam = nnModel->getForwardLayer(0)->getLayerInput()->get(layers::forward::data)->getDimensionSize(0);
    nLayers = forwardLayers->size();

    nSamples = data->getDimensionSize(0);
    if (nSamples < batchSizeParam) { return Status(); }

    /* Get the number of last layers in the network and their indeces */
    lastLayersIndices.reset(new LastLayerIndices(nnModel->getNextLayers().get(), groundTruthCollectionPtr));
    DAAL_CHECK_MALLOC(lastLayersIndices.get() && lastLayersIndices->isValid())

    nLastLayers = lastLayersIndices->nLast(); /* number of last layers in the network */

    /* Create a tensor to pass as an input to the first forward layer in neural network */
    Collection<size_t> sampleSize = data->getDimensions();
    sampleSize[0] = batchSizeParam;
    Status s;
    sample = HomogenTensor<algorithmFPType>::create(sampleSize, Tensor::doNotAllocate, &s);
    DAAL_CHECK_STATUS_VAR(s);


    /* Initialize buffers to manage reading memory operations for the ground truth input tensors */
    groundTruthTensors.reset(nLastLayers);
    DAAL_CHECK_MALLOC(groundTruthTensors.get())

    /* Create tensors to pass as input ground truth to the loss layers in neural network */
    sampleGroundTruthCollection.reset(nLastLayers);
    DAAL_CHECK_MALLOC(sampleGroundTruthCollection.get())

    for (size_t i = 0; i < nLastLayers; i++)
    {
        TensorPtr groundTruthTensor = Tensor::cast((*groundTruthCollectionPtr)[lastLayersIndices->tensorIndex(i)]);
        Collection<size_t> sampleGroundTruthSize = groundTruthTensor->getDimensions();
        sampleGroundTruthSize[0] = batchSizeParam;
        HomogenTensorPtr sampleGroundTruth = HomogenTensor<algorithmFPType>::create(sampleGroundTruthSize, Tensor::doNotAllocate, &s);
        DAAL_CHECK_STATUS_VAR(s);

        sampleGroundTruthCollection[i] = sampleGroundTruth;

        size_t layerId = lastLayersIndices->layerIndex(i);
        loss::forward::Batch *lossLayer = static_cast<loss::forward::Batch *>(forwardLayers->get(layerId).get());
        loss::forward::Input *lossInput = static_cast<loss::forward::Input *>(lossLayer->getLayerInput());
        lossInput->set(loss::forward::groundTruth, sampleGroundTruth);
        lossLayer->getLayerResult()->setResultForBackward(lossInput);
    }
    return s;
}

template<typename algorithmFPType, CpuType cpu>
Status TrainingKernelBase<algorithmFPType, cpu>::computeBase(
    Tensor *data, Model *nnModel,
    const KeyValueDataCollectionPtr &groundTruthCollectionPtr)
{
    ForwardLayersPtr forwardLayers = nnModel->getForwardLayers();
    BackwardLayersPtr backwardLayers = nnModel->getBackwardLayers();

    forward::Input *firstForwardInput = forwardLayers->get(0)->getLayerInput();
    forward::ResultPtr firstForwardResult = forwardLayers->get(0)->getLayerResult();

    firstForwardInput->set(forward::data, sample);
    firstForwardResult->setResultForBackward(firstForwardInput);

    /* Buffer that manages reading memory operations for the input data tensor */
    ReadSubtensor<algorithmFPType, cpu> dataSubtensor(data, 0, 0, 0, 0);

    for (size_t i = 0; i < nLastLayers; i++)
    {
        TensorPtr groundTruthTensor = Tensor::cast((*groundTruthCollectionPtr)[lastLayersIndices->tensorIndex(i)]);
        groundTruthTensors[i].set(*groundTruthTensor, 0, 0, 0, 0);
    }

    size_t maxIterations = getMaxIterations(nSamples, batchSizeParam);

    Status s;
    for(size_t i = 0; i < maxIterations * batchSizeParam; i += batchSizeParam)
    {
        /* Update weights and biases of the network */
        dataSubtensor.next(0, 0, i, batchSizeParam);
        DAAL_CHECK_BLOCK_STATUS(dataSubtensor)
        sample->setArray(const_cast<algorithmFPType *>(dataSubtensor.get()));

        for (size_t j = 0; j < nLastLayers; j++)
        {
            HomogenTensorPtr sampleGroundTruth = HomogenTensor<algorithmFPType>::cast(sampleGroundTruthCollection[j]);
            groundTruthTensors[j].next(0, 0, i, batchSizeParam);
            DAAL_CHECK_BLOCK_STATUS(groundTruthTensors[j])
            sampleGroundTruth->setArray(const_cast<algorithmFPType *>(groundTruthTensors[j].get()));
        }

        /* Forward pass through the neural network */
        for(size_t layerId = 0; layerId < nLayers; layerId++)
        {
            layers::forward::LayerIfacePtr forwardLayer = forwardLayers->get(layerId);
            DAAL_CHECK_STATUS(s, processLayerErrors(layerId, forwardLayer->computeNoThrow()))
        }

        /* Backward pass through the neural network */
        for(int layerId = nLayers - 1; layerId >= 0; layerId--)
        {
            layers::backward::LayerIfacePtr backwardLayer = backwardLayers->get(layerId);
            DAAL_CHECK_STATUS(s, processLayerErrors(layerId, backwardLayer->computeNoThrow()))
        }

        /* Update weights and biases of the network */
        DAAL_CHECK_STATUS(s, updateWeights(*nnModel))
    }
    return s;
}

template<typename algorithmFPType, CpuType cpu>
Status TrainingKernelBase<algorithmFPType, cpu>::resetBase()
{
    lastLayersIndices.reset();
    sampleGroundTruthCollection.reset(0);
    groundTruthTensors.reset(0);
    sample.reset();
    return Status();
}


} // namespace internal
} // namespace feedforward
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
