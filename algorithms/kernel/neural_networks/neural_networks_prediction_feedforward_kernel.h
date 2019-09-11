/* file: neural_networks_prediction_feedforward_kernel.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

//++
//  Declaration of template function that calculate neural networks.
//--


#ifndef __NEURAL_NETWORKS_PREDICTION_FEEDFORWARD_KERNEL_H__
#define __NEURAL_NETWORKS_PREDICTION_FEEDFORWARD_KERNEL_H__

#include "neural_networks/neural_networks_prediction.h"
#include "neural_networks/neural_networks_types.h"
#include "neural_networks/neural_networks_prediction_types.h"

#include "kernel.h"
#include "homogen_tensor.h"
#include "neural_networks_feedforward.h"

#include "service_tensor.h"
#include "service_unique_ptr.h"
#include "service_numeric_table.h"

using namespace daal::data_management;
using namespace daal::services;
using namespace daal::internal;
using namespace daal::algorithms::neural_networks::internal;
using namespace daal::algorithms::neural_networks::layers;

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
/**
 *  \brief Kernel for neural network calculation
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class NeuralNetworksFeedforwardPredictionKernel : public Kernel
{
public:
    services::Status compute(const Input *input, Result *result);
    services::Status initialize(const Input *input, const neural_networks::prediction::Parameter *parameter, Result *result);
    services::Status reset();

private:
    size_t nLastLayers;
    size_t nLayers;
    size_t nSamples;
    size_t batchSize;
    UniquePtr<LastLayerIndices, cpu> lastLayersIndices;
    SharedPtr<HomogenTensor<algorithmFPType> > sample;
    TArray<ReadSubtensor<algorithmFPType, cpu>, cpu> lastLayerResults;
    TArray<WriteOnlySubtensor<algorithmFPType, cpu>, cpu> predictions;
};

} // namespace daal::internal
} // namespace prediction
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
