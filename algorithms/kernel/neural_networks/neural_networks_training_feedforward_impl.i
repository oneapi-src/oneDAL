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

#include "service_numeric_table.h"

using namespace daal::internal;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
using namespace layers;
namespace training
{
namespace internal
{
/**
 *  \brief Kernel for Neural Network training
 */
template<typename algorithmFPType, Method method, CpuType cpu>
void NeuralNetworksFeedforwardTrainingKernel<algorithmFPType, method, cpu>::compute(
    const Input *input, const neural_networks::training::Parameter *parameter, Result *result)
{
    using namespace optimization_solver;
    using namespace optimization_solver::internal;

    SharedPtr<Model> nnModel = result->get(model);
    SharedPtr<ForwardLayers> forwardLayers = nnModel->getForwardLayers();
    SharedPtr<BackwardLayers> backwardLayers = nnModel->getBackwardLayers();
    size_t nLayers = forwardLayers->size();

    SharedPtr<Tensor> data = input->get(training::data);
    size_t nSamples = data->getDimensions().get(0);

    SharedPtr<Tensor> probabilitiesTensor = forwardLayers->get(nLayers - 1)->getLayerResult()->get(forward::value);
    size_t probabilitiesTensorSize = probabilitiesTensor->getSize();
    size_t probabilitiesNRows = probabilitiesTensor->getDimensionSize(0);
    size_t probabilitiesNColumns = probabilitiesTensorSize / probabilitiesNRows;
    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > probabilitiesTable(
        new HomogenNumericTableCPU<algorithmFPType, cpu>(probabilitiesNColumns, probabilitiesNRows));
    algorithmFPType *probabilitiesArray = probabilitiesTable->getArray();
    size_t probabilitiesArraySize = probabilitiesTensorSize * sizeof(algorithmFPType);

    SharedPtr<Tensor> objectiveFunctionGradientTensor = backwardLayers->get(nLayers - 1)->getLayerInput()->get(backward::inputGradient);
    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > objectiveFunctionGradientTable(
        new HomogenNumericTableCPU<algorithmFPType, cpu>(probabilitiesNColumns, probabilitiesNRows));
    algorithmFPType *objectiveFunctionGradientArray = objectiveFunctionGradientTable->getArray();

    SharedPtr<optimization_solver::internal::precomputed::Batch<algorithmFPType> > precomputed(
        new optimization_solver::internal::precomputed::Batch<algorithmFPType>());
    DataCollectionPtr precomputedCollection = DataCollectionPtr(new DataCollection(3));
    SharedPtr<optimization_solver::internal::precomputed::Result> precomputedResult =
        SharedPtr<optimization_solver::internal::precomputed::Result> (new optimization_solver::internal::precomputed::Result());
    precomputedResult->set(objective_function::resultCollection, precomputedCollection);
    precomputed->setResult(precomputedResult);

    SharedPtr<HomogenNumericTable<algorithmFPType> > nIterations(new HomogenNumericTableCPU<algorithmFPType, cpu>(1, 1));
    algorithmFPType *nIterationsArray = nIterations->getArray();
    nIterationsArray[0] = 0;

    SharedPtr<iterative_solver::Batch> iterativeSolverAlgorithm = parameter->optimizationSolver;

    iterativeSolverAlgorithm->parameter->function = precomputed;
    iterativeSolverAlgorithm->parameter->nIterations = 1;

    iterativeSolverAlgorithm->createResult();
    SharedPtr<iterative_solver::Result> iterativeSolverResult = iterativeSolverAlgorithm->getResult();
    iterativeSolverResult->set(iterative_solver::nIterations, nIterations);

    forward::Input *firstForwardInput = forwardLayers->get(0)->getLayerInput();
    SharedPtr<forward::Result> firstForwardResult = forwardLayers->get(0)->getLayerResult();

    bool oneTableForAllWeights = nnModel->getWeightsAndBiasesStorageStatus();
    if (oneTableForAllWeights)
    {
        NumericTablePtr weightsAndBiases = nnModel->getWeightsAndBiases();
        NumericTablePtr weightsAndBiasesDerivatives = nnModel->getWeightsAndBiasesDerivatives();
        precomputedCollection->get(objective_function::gradientIdx) = weightsAndBiasesDerivatives;
        iterativeSolverAlgorithm->input->set(iterative_solver::inputArgument, weightsAndBiases);
        iterativeSolverResult->set(iterative_solver::minimum, weightsAndBiases);
    }

    size_t batchSize = parameter->batchSize;
    SharedPtr<Tensor> groundTruth = input->get(training::groundTruth);
    size_t groundTruthTableNColumns = groundTruth->getSize() / groundTruth->getDimensionSize(0);
    SharedPtr<HomogenNumericTableCPU<algorithmFPType, cpu> > groundTruthTable(new HomogenNumericTableCPU<algorithmFPType, cpu>(
            NULL, groundTruthTableNColumns, batchSize));

    Collection<size_t> sampleSize = data->getDimensions();
    Collection<size_t> sampleGroundTruthSize = groundTruth->getDimensions();
    sampleSize[0] = batchSize;
    sampleGroundTruthSize[0] = batchSize;
    SharedPtr<HomogenTensor<algorithmFPType> > sample(new HomogenTensor<algorithmFPType>(sampleSize, Tensor::notAllocate));
    SharedPtr<HomogenTensor<algorithmFPType> > sampleGroundTruth(new HomogenTensor<algorithmFPType>(sampleGroundTruthSize, Tensor::notAllocate));

    firstForwardInput->set(forward::data, sample);
    firstForwardResult->setResultForBackward(firstForwardInput);

    services::Collection<layers::NextLayers> *nextLayers = nnModel->getNextLayers().get();
    for(size_t layerId = 0; layerId < nLayers; layerId++)
    {
        if (nextLayers->get(layerId).size() == 0)
        {
            loss::forward::Input *lossInput = static_cast<loss::forward::Input *>(forwardLayers->get(layerId)->getLayerInput());
            lossInput->set(loss::forward::groundTruth, sampleGroundTruth);
            forwardLayers->get(layerId)->getLayerResult()->setResultForBackward(lossInput);
        }
    }

    SubtensorDescriptor<algorithmFPType> sampleSubtensor, sampleGroundTruthSubtensor;
    for(size_t i = 0; i < nSamples; i += batchSize)
    {
        data->getSubtensor(0, 0, i, batchSize, readOnly, sampleSubtensor);
        groundTruth->getSubtensor(0, 0, i, batchSize, readOnly, sampleGroundTruthSubtensor);
        sample->setArray(sampleSubtensor.getPtr());
        sampleGroundTruth->setArray(sampleGroundTruthSubtensor.getPtr());

        for(size_t layerId = 0; layerId < nLayers; layerId++)
        {
            forwardLayers->get(layerId)->computeNoThrow();
            if(forwardLayers->get(layerId)->getErrors()->size() != 0)
            {
                groundTruth->releaseSubtensor(sampleGroundTruthSubtensor);
                data->releaseSubtensor(sampleSubtensor);
                this->_errors->add(forwardLayers->get(layerId)->getErrors()->getErrors());
                return;
            }
        }

        for(int layerId = nLayers - 1; layerId >= 0; layerId--)
        {
            backwardLayers->get(layerId)->computeNoThrow();
            if(backwardLayers->get(layerId)->getErrors()->size() != 0)
            {
                groundTruth->releaseSubtensor(sampleGroundTruthSubtensor);
                data->releaseSubtensor(sampleSubtensor);
                this->_errors->add(backwardLayers->get(layerId)->getErrors()->getErrors());
                return;
            }
        }

        if (oneTableForAllWeights)
        {
            iterativeSolverAlgorithm->computeNoThrow();
            if(iterativeSolverAlgorithm->getErrors()->size() != 0)
            {
                groundTruth->releaseSubtensor(sampleGroundTruthSubtensor);
                data->releaseSubtensor(sampleSubtensor);
                this->_errors->add(iterativeSolverAlgorithm->getErrors()->getErrors());
                return;
            }
        }
        else
        {
            for(size_t layerId = 0; layerId < nLayers; layerId++)
            {
                NumericTablePtr weightsAndBiases = nnModel->getWeightsAndBiases(layerId);
                if (weightsAndBiases)
                {
                    NumericTablePtr weightsAndBiasesDerivatives = nnModel->getWeightsAndBiasesDerivatives(layerId);
                    precomputedCollection->get(objective_function::gradientIdx) = weightsAndBiasesDerivatives;
                    iterativeSolverAlgorithm->input->set(iterative_solver::inputArgument, weightsAndBiases);
                    iterativeSolverResult->set(iterative_solver::minimum, weightsAndBiases);

                    iterativeSolverAlgorithm->computeNoThrow();
                    if(iterativeSolverAlgorithm->getErrors()->size() != 0)
                    {
                        groundTruth->releaseSubtensor(sampleGroundTruthSubtensor);
                        data->releaseSubtensor(sampleSubtensor);
                        this->_errors->add(iterativeSolverAlgorithm->getErrors()->getErrors());
                        return;
                    }

                    nnModel->setWeightsAndBiases(layerId, weightsAndBiases);
                }
            }
        }
        groundTruth->releaseSubtensor(sampleGroundTruthSubtensor);
        data->releaseSubtensor(sampleSubtensor);
    }
}

} // namespace internal
} // namespace feedforward
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
