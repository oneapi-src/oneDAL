/* file: neural_networks_training_feedforward_kernel.h */
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
#include "../objective_function/precomputed/precomputed_batch.h"
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
public:
    TrainingKernelBase() : learnableLayerIndices(NULL),
                           lastLayersIndices(NULL),
                           solvers(NULL),
                           sampleGroundTruthCollection(NULL),
                           groundTruthTensors(NULL) {}
protected:
    void initializeBase(Tensor* data, Model *nnModel, const neural_networks::training::Parameter *parameter,
                        KeyValueDataCollectionPtr groundTruthCollectionPtr, bool isBatch);
    void computeBase(Tensor *data, Model *model, KeyValueDataCollectionPtr groundTruthCollectionPtr);
    void resetBase();
private:
    size_t nSolvers;
    size_t batchSizeParam;
    size_t nLastLayers;
    size_t nLayers;
    size_t nSamples;
    bool isBatch;
    bool oneTableForAllWeights;
    LearnableLayerIndices* learnableLayerIndices;
    LastLayerIndices* lastLayersIndices;
    Solver<algorithmFPType> *solvers;
    HomogenTensorPtr sample;
    HomogenTensorPtr *sampleGroundTruthCollection;
    ReadSubtensor<algorithmFPType, cpu> *groundTruthTensors;
};

/**
 * \brief Kernel for neural network calculation in batch mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelBatch : public TrainingKernelBase<algorithmFPType, cpu>
{
    using TrainingKernelBase<algorithmFPType, cpu>::initializeBase;
    using TrainingKernelBase<algorithmFPType, cpu>::computeBase;
    using TrainingKernelBase<algorithmFPType, cpu>::resetBase;
public:
    void initialize(Tensor* data, Model* nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr,
                    const neural_networks::training::Parameter *parameter);
    void compute(Tensor* data, Model* nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr);
    void reset();
};

/**
 * \brief Kernel for neural network calculation in batch mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelDistributed : public TrainingKernelBase<algorithmFPType, cpu>
{
    using TrainingKernelBase<algorithmFPType, cpu>::initializeBase;
    using TrainingKernelBase<algorithmFPType, cpu>::computeBase;
    using TrainingKernelBase<algorithmFPType, cpu>::resetBase;
public:
    void initialize(Tensor* data, Model* nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr,
                    const neural_networks::training::Parameter *parameter);
    void compute(Tensor* data, Model* nnModel, KeyValueDataCollectionPtr groundTruthCollectionPtr,
                 PartialResult *partialResult, const neural_networks::training::Parameter *parameter);
    void reset();
};

/**
 * \brief Kernel for neural network calculation in batch mode
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class TrainingKernelDistributedStep2 : public Kernel
{
public:
    void compute(KeyValueDataCollection* collection, const neural_networks::training::Parameter *parameter, Model* nnModel);
};



} // namespace internal
} // namespace training
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
