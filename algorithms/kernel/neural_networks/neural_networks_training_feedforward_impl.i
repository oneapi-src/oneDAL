/* file: neural_networks_training_feedforward_impl.i */
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
/**
 *  \brief Kernel for Neural Network training in batch processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void TrainingKernelBatch<algorithmFPType, method, cpu>::initialize(Tensor* data, Model* nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr,
                                                const neural_networks::training::Parameter *parameter)
{
    bool isBatch = true;
    initializeBase(data, nnModel, parameter, groundTruthCollectionPtr, isBatch);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void TrainingKernelBatch<algorithmFPType, method, cpu>::compute(Tensor* data, Model* nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr)
{
    computeBase(data, nnModel, groundTruthCollectionPtr);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void TrainingKernelBatch<algorithmFPType, method, cpu>::reset()
{
    resetBase();
}

/**
 *  \brief Kernel for Neural Network training in distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void TrainingKernelDistributed<algorithmFPType, method, cpu>::initialize(Tensor* data, Model* nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr,
                                                                         const neural_networks::training::Parameter *parameter)
{
    bool isBatch = false;
    initializeBase(data, nnModel, parameter, groundTruthCollectionPtr, isBatch);
}

template<typename algorithmFPType, Method method, CpuType cpu>
void TrainingKernelDistributed<algorithmFPType, method, cpu>::compute(Tensor* data, Model* nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr,
                                                                      PartialResult *partialResult, const neural_networks::training::Parameter *parameter)
{
    computeBase(data, nnModel, groundTruthCollectionPtr);

    partialResult->set(derivatives, nnModel->getWeightsAndBiasesDerivatives());

    WriteRows<algorithmFPType, cpu> batchSizeBlock(*(partialResult->get(batchSize)), 0, 1);
    algorithmFPType* batchSizeArray = batchSizeBlock.get();
    batchSizeArray[0] = parameter->batchSize;
}

template<typename algorithmFPType, Method method, CpuType cpu>
void TrainingKernelDistributed<algorithmFPType, method, cpu>::reset()
{
    resetBase();
}

/**
 *  \brief Kernel for Neural Network training in distributed processing mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void TrainingKernelDistributedStep2<algorithmFPType, method, cpu>::compute(KeyValueDataCollection* collection,
                                                                           const neural_networks::training::Parameter *parameter, Model* nnModel)
{
    using namespace optimization_solver;
    using namespace optimization_solver::internal;

    size_t nPartialResults = collection->size();

    SharedPtr<NumericTable> weightsAndBiasesDerivatives;
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

        SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > fullDerivative(
            new HomogenNumericTableCPU<algorithmFPType, cpu>(1, derivSize));

        weightsAndBiasesDerivatives = fullDerivative;
        algorithmFPType* derData = fullDerivative->getArray();
        algorithmFPType sum = 0;

        ReadRows<algorithmFPType, cpu> pDerRows(partialDerivative.get(), 0, derivSize);
        const algorithmFPType* pDerData = pDerRows.get();

        ReadRows<algorithmFPType, cpu> batchSizeBlock(batchSize.get(), 0, 1);
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
            const algorithmFPType* pDerData = pDerRows.get();

            ReadRows<algorithmFPType, cpu> batchSizeBlock(batchSize.get(), 0, 1);
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
    solver.init(parameter->optimizationSolver);
    SharedPtr<KernelErrorCollection> solverErrors = solver.updateWeightsAndBiases(
                    nnModel->getWeightsAndBiases(), weightsAndBiasesDerivatives);
    if(solverErrors->size() != 0)
    {
        this->_errors->add(solverErrors);
        return;
    }
    nnModel->setWeightsAndBiases(solver.getMinimum());
}

/**
 *  \brief Kernel for Neural Network training
 */
template<typename algorithmFPType, CpuType cpu>
void TrainingKernelBase<algorithmFPType, cpu>::initializeBase(Tensor *data, Model *nnModel, const neural_networks::training::Parameter *parameter,
                                                              KeyValueDataCollectionPtr groundTruthCollectionPtr, bool _isBatch)
{
    SharedPtr<ForwardLayers> forwardLayers = nnModel->getForwardLayers();
    batchSizeParam = parameter->batchSize;
    nLayers = forwardLayers->size();
    isBatch = _isBatch;

    nSamples = data->getDimensionSize(0);
    if (nSamples < batchSizeParam) { return; }

    learnableLayerIndices = new LearnableLayerIndices(forwardLayers.get());
    if (learnableLayerIndices->getError())
    {
        resetBase();
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    /* Get the number of last layers in the network and their indeces */
    lastLayersIndices = new LastLayerIndices(nnModel->getNextLayers().get(), groundTruthCollectionPtr);
    if (lastLayersIndices->getError())
    {
        resetBase();
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }
    nLastLayers = lastLayersIndices->nLast(); /* number of last layers in the network */

    nSolvers = 0;
    oneTableForAllWeights = nnModel->getWeightsAndBiasesStorageStatus();
    if(isBatch)
    {
        if (oneTableForAllWeights)
        {
            nSolvers = 1;
        }
        else
        {
            nSolvers = learnableLayerIndices->nLearnable();
        }

        solvers = new Solver<algorithmFPType>[nSolvers];
        if (!solvers)
        {
            resetBase();
            this->_errors->add(ErrorMemoryAllocationFailed);
            return;
        }

        for (size_t i = 0; i < nSolvers; i++)
        {
            solvers[i].init(parameter->optimizationSolver);
        }
    }

    /* Create a tensor to pass as an input to the first forward layer in neural network */
    Collection<size_t> sampleSize = data->getDimensions();
    sampleSize[0] = batchSizeParam;
    sample.reset(new HomogenTensor<algorithmFPType>(sampleSize, Tensor::notAllocate));

    /* Initialize buffers to manage reading memory operations for the ground truth input tensors */
    groundTruthTensors = new ReadSubtensor<algorithmFPType, cpu>[nLastLayers];

    /* Create tensors to pass as input ground truth to the loss layers in neural network */
    sampleGroundTruthCollection = new HomogenTensorPtr[nLastLayers];
    if (!sampleGroundTruthCollection || !groundTruthTensors)
    {
        resetBase();
        this->_errors->add(ErrorMemoryAllocationFailed);
        return;
    }

    for (size_t i = 0; i < nLastLayers; i++)
    {
        TensorPtr groundTruthTensor = Tensor::cast((*groundTruthCollectionPtr)[lastLayersIndices->tensorIndex(i)]);
        Collection<size_t> sampleGroundTruthSize = groundTruthTensor->getDimensions();
        sampleGroundTruthSize[0] = batchSizeParam;
        HomogenTensorPtr sampleGroundTruth(new HomogenTensor<algorithmFPType>(sampleGroundTruthSize, Tensor::notAllocate));
        sampleGroundTruthCollection[i] = sampleGroundTruth;

        size_t layerId = lastLayersIndices->layerIndex(i);
        loss::forward::Batch *lossLayer = static_cast<loss::forward::Batch *>(forwardLayers->get(layerId).get());
        loss::forward::Input *lossInput = static_cast<loss::forward::Input *>(lossLayer->getLayerInput());
        lossInput->set(loss::forward::groundTruth, sampleGroundTruth);
        lossLayer->getLayerResult()->setResultForBackward(lossInput);
    }
}

template<typename algorithmFPType, CpuType cpu>
void TrainingKernelBase<algorithmFPType, cpu>::computeBase(Tensor *data, Model *nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr)
{
    using namespace optimization_solver;
    using namespace optimization_solver::internal;

    SharedPtr<ForwardLayers> forwardLayers = nnModel->getForwardLayers();
    SharedPtr<BackwardLayers> backwardLayers = nnModel->getBackwardLayers();

    forward::Input *firstForwardInput = forwardLayers->get(0)->getLayerInput();
    SharedPtr<forward::Result> firstForwardResult = forwardLayers->get(0)->getLayerResult();

    firstForwardInput->set(forward::data, sample);
    firstForwardResult->setResultForBackward(firstForwardInput);

    /* Buffer that manages reading memory operations for the input data tensor */
    ReadSubtensor<algorithmFPType, cpu> dataSubtensor(data, 0, 0, 0, 0);

    for (size_t i = 0; i < nLastLayers; i++)
    {
        TensorPtr groundTruthTensor = Tensor::cast((*groundTruthCollectionPtr)[lastLayersIndices->tensorIndex(i)]);
        groundTruthTensors[i].set(*groundTruthTensor, 0, 0, 0, 0);
    }

    for(size_t i = 0; i < nSamples - batchSizeParam + 1; i += batchSizeParam)
    {
        /* Update weights and biases of the network */
        sample->setArray(const_cast<algorithmFPType *>(dataSubtensor.next(0, 0, i, batchSizeParam)));
        for (size_t j = 0; j < nLastLayers; j++)
        {
            HomogenTensorPtr sampleGroundTruth = HomogenTensor<algorithmFPType>::cast(sampleGroundTruthCollection[j]);
            sampleGroundTruth->setArray(const_cast<algorithmFPType *>(groundTruthTensors[j].next(0, 0, i, batchSizeParam)));
        }

        /* Forward pass through the neural network */
        for(size_t layerId = 0; layerId < nLayers; layerId++)
        {
            layers::forward::LayerIfacePtr forwardLayer = forwardLayers->get(layerId);
            forwardLayer->computeNoThrow();
            if (!processLayerErrors(layerId, forwardLayer->getErrors()->getErrors(), this->_errors))
            {
                resetBase();
                return;
            }
        }

        /* Backward pass through the neural network */
        for(int layerId = nLayers - 1; layerId >= 0; layerId--)
        {
            layers::backward::LayerIfacePtr backwardLayer = backwardLayers->get(layerId);
            backwardLayer->computeNoThrow();
            if (!processLayerErrors(layerId, backwardLayer->getErrors()->getErrors(), this->_errors))
            {
                resetBase();
                return;
            }
        }

        /* Update weights and biases of the network */
        if (isBatch)
        {
            if (oneTableForAllWeights)
            {
                Solver<algorithmFPType> &solver = solvers[0];
                SharedPtr<KernelErrorCollection> solverErrors = solver.updateWeightsAndBiases(
                    nnModel->getWeightsAndBiases(), nnModel->getWeightsAndBiasesDerivatives());
                if(solverErrors->size() != 0)
                {
                    resetBase();
                    this->_errors->add(solverErrors);
                    return;
                }
                nnModel->setWeightsAndBiases(solver.getMinimum());
            }
            else
            {
                for(size_t i = 0; i < nSolvers; i++)
                {
                    size_t layerId = learnableLayerIndices->layerIndex(i);
                    Solver<algorithmFPType> &solver = solvers[i];

                    SharedPtr<KernelErrorCollection> solverErrors = solver.updateWeightsAndBiases(
                        nnModel->getWeightsAndBiases(layerId), nnModel->getWeightsAndBiasesDerivatives(layerId));

                    if(solverErrors->size() != 0)
                    {
                        resetBase();
                        this->_errors->add(solverErrors);
                        return;
                    }

                    nnModel->setWeightsAndBiases(layerId, solver.getMinimum());
                }
            }
        }
    }
}

template<typename algorithmFPType, CpuType cpu>
void TrainingKernelBase<algorithmFPType, cpu>::resetBase()
{
    if(solvers)                     { delete [] solvers; solvers = NULL; }
    if(learnableLayerIndices)       { delete learnableLayerIndices; learnableLayerIndices = NULL; }
    if(lastLayersIndices)           { delete lastLayersIndices; lastLayersIndices = NULL; }
    if(sampleGroundTruthCollection) { delete [] sampleGroundTruthCollection; sampleGroundTruthCollection = NULL; }
    if(groundTruthTensors)          { delete [] groundTruthTensors; groundTruthTensors = NULL; }
    sample.reset();

}


} // namespace internal
} // namespace feedforward
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
