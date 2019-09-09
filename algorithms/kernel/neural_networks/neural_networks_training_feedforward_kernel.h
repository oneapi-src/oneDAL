/* file: neural_networks_training_feedforward_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "neural_networks_feedforward.h"
#include "neural_networks_training_feedforward.h"
#include "neural_networks/neural_networks_training.h"
#include "neural_networks/neural_networks_training_types.h"
#include "neural_networks/layers/loss/loss_layer_forward_types.h"
#include "optimization_solver/objective_function/precomputed_batch.h"
#include "optimization_solver/iterative_solver/iterative_solver_batch.h"
#include "optimization_solver/iterative_solver/iterative_solver_types.h"

#include "kernel.h"
#include "numeric_table.h"

#include "service_tensor.h"
#include "service_unique_ptr.h"
#include "service_numeric_table.h"

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
public:
    typedef SharedPtr<HomogenTensor<algorithmFPType>> HomogenTensorPtr;

protected:
    Status initializeBase(Tensor* data, Model *nnModel,
                          const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
                          const neural_networks::training::Parameter *parameter);

    Status computeBase(Tensor *data, Model *model,
                       const KeyValueDataCollectionPtr &groundTruthCollectionPtr);

    Status resetBase();

    virtual Status updateWeights(Model &nnModel) = 0;
    virtual size_t getMaxIterations(size_t nSamples, size_t batchSizeParam) const = 0;

private:
    size_t batchSizeParam;
    size_t nLastLayers;
    size_t nLayers;
    size_t nSamples;

    HomogenTensorPtr sample;
    UniquePtr<LastLayerIndices, cpu> lastLayersIndices;
    TArray<HomogenTensorPtr, cpu> sampleGroundTruthCollection;
    TArray<ReadSubtensor<algorithmFPType, cpu>, cpu> groundTruthTensors;
};

/**
 * \brief Kernel for neural network calculation in batch mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelBatch : public TrainingKernelBase<algorithmFPType, cpu>
{
public:
    Status initialize(Tensor* data, Model* nnModel,
                      const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
                      const neural_networks::training::Parameter *parameter);

    Status compute(Tensor* data, Model* nnModel,
                   const KeyValueDataCollectionPtr &groundTruthCollectionPtr);

    Status reset();

protected:
    virtual Status updateWeights(Model &nnModel) DAAL_C11_OVERRIDE;
    virtual size_t getMaxIterations(size_t nSamples, size_t batchSizeParam) const DAAL_C11_OVERRIDE;

private:
    bool oneTableForAllWeights;
    UniquePtr<LearnableLayerIndices, cpu> learnableLayerIndices;
    TArray<Solver<algorithmFPType>, cpu> solvers;
};

/**
 * \brief Kernel for neural network calculation in distributed mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelDistributed : public TrainingKernelBase<algorithmFPType, cpu>
{
public:
    Status initialize(Tensor* data, Model* nnModel,
                      const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
                      const neural_networks::training::Parameter *parameter);

    Status compute(Tensor* data, Model* nnModel, const KeyValueDataCollectionPtr &groundTruthCollectionPtr,
                   PartialResult *partialResult, const neural_networks::training::Parameter *parameter);

    Status reset();

protected:
    virtual Status updateWeights(Model &nnModel) DAAL_C11_OVERRIDE;
    virtual size_t getMaxIterations(size_t nSamples, size_t batchSizeParam) const DAAL_C11_OVERRIDE;
};

/**
 * \brief Kernel for neural network calculation in distributed mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelDistributedStep2 : public Kernel
{
public:
    Status compute(KeyValueDataCollection* collection, const Parameter *parameter, Model* nnModel);
};


} // namespace internal
} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
