/* file: spatial_average_pooling2d_layer_backward.h */
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
//  Implementation of the interface for the backward spatial pyramid average two-dimensional (2D) pooling layer
//  in the batch processing mode
//--
*/

#ifndef __SPATIAL_AVERAGE_POOLING2D_LAYER_BACKWARD_H__
#define __SPATIAL_AVERAGE_POOLING2D_LAYER_BACKWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/spatial_pooling2d/spatial_average_pooling2d_layer_types.h"
#include "algorithms/neural_networks/layers/spatial_pooling2d/spatial_average_pooling2d_layer_backward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace spatial_average_pooling2d
{
namespace backward
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * @defgroup spatial_average_pooling2d_backward_batch Batch
 * @ingroup spatial_average_pooling2d_backward
 * @{
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{};
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__BACKWARD__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the backward spatial pyramid average 2D pooling layer.
 *        This class is associated with the \ref backward::interface1::Batch "backward::Batch" class
 *        and supports the method of backward spatial pyramid average 2D pooling layer computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of backward spatial pyramid average 2D pooling layer, double or float
 * \tparam method           Computation method of the layer, spatial_average_pooling2d::Method
 * \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
 */
template<typename algorithmFPType, CpuType cpu>
class DAAL_EXPORT BatchContainer<algorithmFPType, defaultDense, cpu> : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the backward spatial pyramid average 2D pooling layer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the backward spatial pyramid average 2D pooling layer in the batch processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__SPATIAL_AVERAGE_POOLING2D__BACKWARD__BATCH"></a>
 * \brief Provides methods for the backward spatial pyramid average 2D pooling layer in the batch processing mode
 * \n<a href="DAAL-REF-SPATIAL_AVERAGEPOOLING2DBACKWARD-ALGORITHM">Backward spatial pyramid average 2D pooling layer description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the backward  spatial pyramid average 2D pooling layer, double or float
 * \tparam method           Backward spatial pyramid average 2D pooling layer method, spatial_average_pooling2d::Method
 *
 * \par Enumerations
 *      - \ref Method                      Computation methods for the backward spatial pyramid average 2D pooling layer
 *      - \ref backward::InputId           Identifiers of input objects for the backward spatial pyramid average 2D pooling layer
 *      - \ref LayerDataId                 Identifiers of collection in input objects for the backward spatial pyramid average 2D pooling layer
 *      - \ref backward::InputLayerDataId  Identifiers of extra results computed by the forward spatial pyramid average 2D pooling layer
 *      - \ref backward::ResultId          Identifiers of result objects for the backward spatial pyramid average 2D pooling layer
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref interface1::Input "Input" class
 *      - \ref interface1::Result "Result" class
 *      - \ref forward::interface1::Batch "forward::Batch" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public layers::backward::LayerIface
{
public:
    Parameter &parameter; /*!< Backward spatial pyramid average 2D pooling layer parameters */
    Input input;          /*!< Backward spatial pyramid average 2D pooling layer input */

    /**
     * Constructs backward spatial pyramid average 2D pooling layer with the provided parameters
     * \param[in] nDimensions   Number of dimensions in input data tensor
     * \param[in] pyramidHeight The value of pyramid height
     */
    Batch(size_t pyramidHeight, size_t nDimensions) : _defaultParameter(pyramidHeight, nDimensions - 2, nDimensions - 1), parameter(_defaultParameter)
    {
        initialize();
    }

    /**
     * Constructs a backward spatial pyramid average 2D pooling layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(Parameter& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs a backward spatial pyramid average 2D pooling layer by copying input objects
     * and parameters of another backward spatial pyramid average 2D pooling layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the layer
     */
    Batch(const Batch<algorithmFPType, method> &other) :  _defaultParameter(other.parameter), parameter(_defaultParameter)
    {
        initialize();
        input.set(layers::backward::inputGradient,    other.input.get(layers::backward::inputGradient));
        input.set(layers::backward::inputFromForward, other.input.get(layers::backward::inputFromForward));
    }

    /**
     * Returns the method of the layer
     * \return Method of the layer
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the input objects of backward spatial pyramid average 2D pooling layer
     * \return Structure that contains the input objects of backward spatial pyramid average 2D pooling layer
     */
    virtual Input *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains the parameters of the backward spatial pyramid average 2D pooling layer
     * \return Structure that contains the parameters of the backward spatial pyramid average 2D pooling layer
     */
    virtual Parameter *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains result of the backward spatial pyramid average 2D pooling layer
     * \return Structure that contains result of the backward spatial pyramid average 2D pooling layer
     */
    services::SharedPtr<layers::backward::Result> getLayerResult() DAAL_C11_OVERRIDE
    {
        return _result;
    }

    /**
     * Returns the structure that contains the result of the backward spatial pyramid average 2D pooling layer
     * \return Structure that contains the result of the backward spatial pyramid average 2D pooling layer
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward spatial pyramid average 2D pooling layer
     * \param[in] result Structure to store the result of the backward spatial pyramid average 2D pooling layer
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to a newly allocated backward spatial pyramid average 2D pooling layer
     * with a copy of the input objects and parameters for this backward spatial pyramid average 2D pooling layer
     * in the batch processing mode
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the backward spatial pyramid average 2D pooling layer
     */
    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int) method);
        this->_res = this->_result.get();
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in = &input;
        _par = &parameter;
        _result = services::SharedPtr<Result>(new Result());
    }

private:
    services::SharedPtr<Result> _result;
    Parameter _defaultParameter;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;
} // namespace backward
} // namespace spatial_average_pooling2d
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
