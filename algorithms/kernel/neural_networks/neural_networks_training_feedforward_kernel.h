/* file: neural_networks_training_feedforward_kernel.h */
/*******************************************************************************
* Copyright 2014-2017 Intel Corporation
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

//++
//  Declaration of template function that calculate neural networks.
//--


#ifndef __NEURAL_NETWORKS_TRAINING_FEEDFORWARD_KERNEL_H__
#define __NEURAL_NETWORKS_TRAINING_FEEDFORWARD_KERNEL_H__

#include "neural_networks/neural_networks_training.h"
#include "neural_networks/neural_networks_training_types.h"
#include "kernel.h"
#include "numeric_table.h"
#include "service_numeric_table.h"
#include "neural_networks/layers/loss/loss_layer_forward_types.h"
#include "optimization_solver/objective_function/precomputed_batch.h"
#include "optimization_solver/iterative_solver/iterative_solver_batch.h"
#include "optimization_solver/iterative_solver/iterative_solver_types.h"
#include "service_tensor.h"
#include "neural_networks_feedforward.h"
#include "neural_networks_training_feedforward.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::algorithms::neural_networks::internal;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::internal;

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
 * \brief Common kernel for neural network calculation
 */
template<typename algorithmFPType, CpuType cpu>
class TrainingKernelBase : public Kernel
{
    typedef SharedPtr<HomogenTensor<algorithmFPType> > HomogenTensorPtr;
protected:
    services::Status initializeBase(Tensor* data, Model *nnModel, const neural_networks::training::Parameter *parameter,
                        const KeyValueDataCollectionPtr &groundTruthCollectionPtr);
    services::Status computeBase(Tensor *data, Model *model, const KeyValueDataCollectionPtr &groundTruthCollectionPtr);
    services::Status reset();
    virtual services::Status updateWeights(Model &nnModel) = 0;
    virtual size_t getMaxIterations(size_t nSamples, size_t batchSizeParam) const = 0;
private:
    size_t batchSizeParam;
    size_t nLastLayers;
    size_t nLayers;
    size_t nSamples;
    UniquePtr<LastLayerIndices, cpu> lastLayersIndices;
    HomogenTensorPtr sample;
    TArray<HomogenTensorPtr, cpu> sampleGroundTruthCollection;
    TArray<ReadSubtensor<algorithmFPType, cpu>, cpu> groundTruthTensors;
};

/**
 * \brief Kernel for neural network calculation in batch mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelBatch : public TrainingKernelBase<algorithmFPType, cpu>
{
    typedef TrainingKernelBase<algorithmFPType, cpu> super;
    using TrainingKernelBase<algorithmFPType, cpu>::initializeBase;
    using TrainingKernelBase<algorithmFPType, cpu>::computeBase;
    services::Status updateWeights(Model &nnModel) DAAL_C11_OVERRIDE;
    size_t getMaxIterations(size_t nSamples, size_t batchSizeParam) const DAAL_C11_OVERRIDE;
public:
    services::Status initialize(Tensor* data, Model* nnModel, const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
                    const neural_networks::training::Parameter *parameter);
    services::Status compute(Tensor* data, Model* nnModel, const KeyValueDataCollectionPtr &groundTruthCollectionPtr);
    services::Status reset();
private:
    bool oneTableForAllWeights;
    UniquePtr<LearnableLayerIndices, cpu> learnableLayerIndices;
    TArray<Solver<algorithmFPType>, cpu> solvers;
};

/**
 * \brief Kernel for neural network calculation in batch mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelDistributed : public TrainingKernelBase<algorithmFPType, cpu>
{
    typedef TrainingKernelBase<algorithmFPType, cpu> super;
    using TrainingKernelBase<algorithmFPType, cpu>::initializeBase;
    using TrainingKernelBase<algorithmFPType, cpu>::computeBase;
    services::Status updateWeights(Model &nnModel) DAAL_C11_OVERRIDE;
    size_t getMaxIterations(size_t nSamples, size_t batchSizeParam) const DAAL_C11_OVERRIDE;
public:
    services::Status initialize(Tensor* data, Model* nnModel, const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
                    const neural_networks::training::Parameter *parameter);
    services::Status compute(Tensor* data, Model* nnModel, const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
                 PartialResult *partialResult, const neural_networks::training::Parameter *parameter);
    using TrainingKernelBase<algorithmFPType, cpu>::reset;
};

/**
 * \brief Kernel for neural network calculation in batch mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelDistributedStep2 : public Kernel
{
public:
    services::Status compute(KeyValueDataCollection* collection, const neural_networks::training::Parameter *parameter, Model* nnModel);
};



} // namespace internal
} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
