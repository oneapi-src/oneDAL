/* file: spatial_maximum_pooling2d_layer_forward.h */
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
//  Implementation of the interface for the forward spatial pyramid maximum two-dimensional (2D) pooling layer
//  in the batch processing mode
//--
*/

#ifndef __SPATIAL_MAXIMUM_POOLING2D_LAYER_FORWARD_H__
#define __SPATIAL_MAXIMUM_POOLING2D_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/spatial_pooling2d/spatial_maximum_pooling2d_layer_types.h"
#include "algorithms/neural_networks/layers/spatial_pooling2d/spatial_maximum_pooling2d_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_maximum_pooling2d
{
namespace forward
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * @defgroup spatial_maximum_pooling2d_forward_batch Batch
 * @ingroup spatial_maximum_pooling2d_forward
 * @{
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public layers::forward::LayerContainerIfaceImpl
{};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_MAXIMUM_POOLING2D__FORWARD__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the forward spatial pyramid maximum 2D pooling layer.
 *        This class is associated with the \ref forward::interface1::Batch "forward::Batch" class
 *        and supports the method of forward spatial pyramid maximum 2D pooling layer computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of forward spatial pyramid maximum 2D pooling layer, double or float
 * \tparam method           Computation method of the layer, spatial_maximum_pooling2d::Method
 * \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
 */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT BatchContainer<algorithmFPType, defaultDense, cpu> : public layers::forward::LayerContainerIfaceImpl
{
public:
    /**
     * Constructs a container for the forward spatial pyramid maximum 2D pooling layer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the forward spatial pyramid maximum 2D pooling layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_MAXIMUM_POOLING2D__FORWARD__BATCH"></a>
 * \brief Provides methods for the forward spatial pyramid maximum 2D pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-SPATIAL_MAXIMUMPOOLING2DFORWARD-ALGORITHM">Forward spatial pyramid maximum 2D pooling layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the forward spatial pyramid maximum 2D pooling layer, double or float
 * \tparam method           Forward spatial pyramid maximum 2D pooling layer method, spatial_maximum_pooling2d::Method
 *
 * \par Enumerations
 *      - \ref Method                     Computation methods for the forward spatial pyramid maximum 2D pooling layer
 *      - \ref forward::InputId           Identifiers of input objects for the forward spatial pyramid maximum 2D pooling layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward spatial pyramid maximum 2D pooling layer
 *      - \ref forward::ResultLayerDataId Identifiers of extra results computed by the forward spatial pyramid maximum 2D pooling layer
 *      - \ref LayerDataId                Identifiers of collection in result objects for the forward spatial pyramid maximum 2D pooling layer
 *
 * \par References
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public layers::forward::LayerIfaceImpl
{
public:
    typedef layers::forward::LayerIfaceImpl super;

    typedef algorithms::neural_networks::layers::spatial_maximum_pooling2d::forward::Input     InputType;
    typedef algorithms::neural_networks::layers::spatial_maximum_pooling2d::Parameter          ParameterType;
    typedef algorithms::neural_networks::layers::spatial_maximum_pooling2d::forward::Result    ResultType;

    ParameterType &parameter; /*!< Forward spatial pyramid maximum 2D pooling layer \ref interface1::Parameter "parameters" */
    InputType input;          /*!< Forward spatial pyramid maximum 2D pooling layer input */

    /**
     * Constructs forward spatial pyramid maximum 2D pooling layer with the provided parameters
     * \param[in] nDimensions   Number of dimensions in input data tensor
     * \param[in] pyramidHeight The value of pyramid height
     */
    Batch(size_t pyramidHeight, size_t nDimensions) : _defaultParameter(pyramidHeight, nDimensions - 2, nDimensions - 1),parameter(_defaultParameter)
    {
        initialize();
    }

    /**
     * Constructs a forward spatial pyramid maximum 2D pooling layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(ParameterType& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs a forward spatial pyramid maximum 2D pooling layer by copying input objects
     * and parameters of another forward spatial pyramid maximum 2D pooling layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the layer
     */
    Batch(const Batch<algorithmFPType, method> &other) : super(other),
        _defaultParameter(other.parameter), parameter(_defaultParameter), input(other.input)
    {
        initialize();
    }

    /**
     * Returns the method of the layer
     * \return Method of the layer
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the input objects of the forward spatial pyramid maximum 2D pooling layer
     * \return Structure that contains the input objects of the forward spatial pyramid maximum 2D pooling layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains the parameters of the forward spatial pyramid maximum 2D pooling layer
     * \return Structure that contains the parameters of the forward spatial pyramid maximum 2D pooling layer
     */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains result of the forward spatial pyramid maximum 2D pooling layer
     * \return Structure that contains result of the forward spatial pyramid maximum 2D pooling layer
     */
    layers::forward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return getResult();
    }

    /**
     * Returns the structure that contains the result of the forward spatial pyramid maximum 2D pooling layer
     * \return Structure that contains the result of the forward spatial pyramid maximum 2D pooling layer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward spatial pyramid maximum 2D pooling layer
     * \param[in] result Structure to store the result of the forward spatial pyramid maximum 2D pooling layer
     *
     * \return Status of computations
     */
    services::Status setResult(const ResultPtr& result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to a newly allocated forward spatial pyramid maximum 2D pooling layer
     * with a copy of the input objects and parameters for this forward spatial pyramid maximum 2D pooling layer
     * in the batch processing mode
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the forward spatial pyramid maximum 2D pooling layer
     *
     * \return Status of computations
     */
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int) method);
        this->_res = this->_result.get();
        return s;
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_LAYER_CONTAINER(BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }

private:
    ResultPtr _result;
    ParameterType _defaultParameter;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
} // namespace forward
} // namespace spatial_maximum_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
