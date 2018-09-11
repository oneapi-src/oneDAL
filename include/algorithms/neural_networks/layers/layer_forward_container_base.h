/* file: layer_forward_container_base.h */
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
//  Base container of neural network forward layer.
//--
*/

#ifndef __LAYER_FORWARD_CONTAINER_BASE_H__
#define __LAYER_FORWARD_CONTAINER_BASE_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer_forward_types.h"

namespace daal
{
namespace algorithms
{
/**
 * \brief Contains classes for training and prediction using neural network
 */
namespace neural_networks
{
/**
 * \brief Contains classes for neural network layers
 */
namespace layers
{
/**
 * \brief Contains classes for the forward stage of the neural network layer
 */
namespace forward
{
namespace interface1
{
/**
 * @ingroup layers_forward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__LAYERCONTAINERIFACEIMPL"></a>
 * \brief Provides methods of base container for forward layers.
 *        This class is associated with the daal::algorithms::neural_networks::layers::forward::LayerContainerIfaceImpl class
 */
class DAAL_EXPORT LayerContainerIfaceImpl : public AnalysisContainerIface<batch>
{
public:
    LayerContainerIfaceImpl(daal::services::Environment::env *daalEnv = 0) : AnalysisContainerIface<batch>(daalEnv) {}

    /**
     * \copydoc daal::data_management::interface1::SerializationIface::getSerializationTag()
     */
    virtual services::Status setupCompute()
    {
        return services::Status();
    }

    virtual services::Status resetCompute()
    {
        return services::Status();
    }

    /**
     * Allocates weights and biases tensors if they exist
     */
    virtual services::Status allocateInput();

    /**
     * Initializes values of weights and biases
     */
    virtual services::Status initializeInput();

protected:
    virtual services::Status completeInput();
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__FORWARD__ALGORITHMDISPATCHLAYERCONTAINER"></a>
 * \brief Implements a container to dispatch forward layers to cpu-specific implementations.
 *
 *
 * \tparam sse2Container        Implementation for Intel(R) Streaming SIMD Extensions 2 (Intel(R) SSE2)
 * \tparam ssse3Container       Implementation for Supplemental Streaming SIMD Extensions 3 (SSSE3)
 * \tparam sse42Container       Implementation for Intel(R) Streaming SIMD Extensions 42 (Intel(R) SSE42)
 * \tparam avxContainer         Implementation for Intel(R) Advanced Vector Extensions (Intel(R) AVX)
 * \tparam avx2Container        Implementation for Intel(R) Advanced Vector Extensions 2 (Intel(R) AVX2)
 * \tparam avx512_micContainer  Implementation for Intel(R) Xeon Phi(TM) processors/coprocessors based on Intel(R) Advanced Vector
 *                              Extensions 512 (Intel(R) AVX512)
 * \tparam avx512Container      Implementation for Intel(R) Xeon(R) processors based on Intel AVX-512
 */
template<ComputeMode mode,
         typename sse2Container
         DAAL_KERNEL_SSSE3_ONLY(typename ssse3Container)
         DAAL_KERNEL_SSE42_ONLY(typename sse42Container)
         DAAL_KERNEL_AVX_ONLY(typename avxContainer)
         DAAL_KERNEL_AVX2_ONLY(typename avx2Container)
         DAAL_KERNEL_AVX512_mic_ONLY(typename avx512_micContainer)
         DAAL_KERNEL_AVX512_ONLY(typename avx512Container)
         >
class DAAL_EXPORT AlgorithmDispatchLayerContainer : public LayerContainerIfaceImpl
{
public:
    /** Default constructor. Constructs empty container */
    AlgorithmDispatchLayerContainer(daal::services::Environment::env *daalEnv);
    virtual ~AlgorithmDispatchLayerContainer() { delete _cntr; }

    virtual services::Status compute() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_res, this->_par);
        return _cntr->compute();
    }

    virtual services::Status setupCompute() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_res, this->_par);
        return _cntr->setupCompute();
    }

    virtual services::Status resetCompute() DAAL_C11_OVERRIDE
    {
        return _cntr->resetCompute();
    }

    virtual services::Status allocateInput() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_res, this->_par);
        return _cntr->allocateInput();
    }

    virtual services::Status initializeInput() DAAL_C11_OVERRIDE
    {
        _cntr->setArguments(this->_in, this->_res, this->_par);
        return _cntr->initializeInput();
    }
protected:
    LayerContainerIfaceImpl *_cntr;
};

#define __DAAL_ALGORITHM_LAYER_CONTAINER(ContainerTemplate, ...)     \
    layers::forward::AlgorithmDispatchLayerContainer<batch,          \
    ContainerTemplate<__VA_ARGS__, sse2>                             \
    DAAL_KERNEL_SSSE3_CONTAINER(ContainerTemplate, __VA_ARGS__)      \
    DAAL_KERNEL_SSE42_CONTAINER(ContainerTemplate, __VA_ARGS__)      \
    DAAL_KERNEL_AVX_CONTAINER(ContainerTemplate, __VA_ARGS__)        \
    DAAL_KERNEL_AVX2_CONTAINER(ContainerTemplate, __VA_ARGS__)       \
    DAAL_KERNEL_AVX512_mic_CONTAINER(ContainerTemplate, __VA_ARGS__) \
    DAAL_KERNEL_AVX512_CONTAINER(ContainerTemplate, __VA_ARGS__)>

/** @} */

} // namespace interface1

using interface1::AlgorithmDispatchLayerContainer;
using interface1::LayerContainerIfaceImpl;

} // namespace forward
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
