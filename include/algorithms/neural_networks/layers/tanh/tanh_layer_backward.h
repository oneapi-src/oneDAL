/* file: tanh_layer_backward.h */
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
//  Implementation of the interface for the backward hyperbolic tangent layer
//  in the batch processing mode
//--
*/

#ifndef __TANH_LAYER_BACKWARD_H__
#define __TANH_LAYER_BACKWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/tanh/tanh_layer_types.h"
#include "algorithms/neural_networks/layers/tanh/tanh_layer_backward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the hyperbolic tangent layer
 */
namespace tanh
{
/**
 * \brief Contains classes for the backward hyperbolic tangent layer
 */
namespace backward
{
namespace interface1
{
/**
 * @defgroup tanh_layers_backward_batch Batch
 * @ingroup tanh_layers_backward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__BACKWARD__BATCHCONTAINER"></a>
* \brief Provides methods to run implementations of the of the backward tanh layer
*        This class is associated with the daal::algorithms::neural_networks::layers::tanh::backward::Batch class
*        and supports the method of backward tanh layer computation in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations of backward tanh layer, double or float
* \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::tanh::Method
* \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
*/
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
    * Constructs a container for the backward hyperbolic tangent layer with a specified environment
    * in the batch processing mode
    * \param[in] daalEnv   Environment object
    */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the backward hyperbolic tangent layer in the batch processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__TANH__BACKWARD__BATCH"></a>
 * \brief Computes the results of the backward hyperbolic tangent in the batch processing mode
 * \n<a href="DAAL-REF-TANHBACKWARD-ALGORITHM">Backward hyperbolic tangent layer description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the backward hyperbolic tangent layer, double or float
 * \tparam method           The backward hyperbolic tangent layer computation method, \ref Method
 * \par Enumerations
 *      - \ref Method                      Computation methods for the backward hyperbolic tangent layer
 *      - \ref backward::InputId           Identifiers of input objects for the backward hyperbolic tangent layer
 *      - \ref LayerDataId                 Identifiers of collection in input objects for the hyperbolic tangent layer
 *      - \ref backward::InputLayerDataId  Identifiers of extra results computed by the forward hyperbolic tangent layer
 *      - \ref backward::ResultId          Identifiers of result objects for the backward hyperbolic tangent layer
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref forward::interface1::Batch "forward::Batch" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public layers::backward::LayerIface
{
public:
    Parameter &parameter; /*!< tanh layer parameters structure */
    Input input;         /*!< %Input objects of the layer */

    /** Default constructor */
    Batch() : parameter(_defaultParameter)
    {
        initialize();
    }

    /**
     * Constructs a backward tanh layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(Parameter& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs the backward hyperbolic tangent layer by copying input objects of
     * another backward hyperbolic tangent layer in the batch processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  of the backward hyperbolic tangent layer
     */
    Batch(const Batch<algorithmFPType, method> &other) : _defaultParameter(other.parameter), parameter(_defaultParameter)
    {
        initialize();
        input.set(layers::backward::inputGradient, other.input.get(layers::backward::inputGradient));
        input.set(layers::backward::inputFromForward, other.input.get(layers::backward::inputFromForward));
    }

    /**
    * Returns method of the backward hyperbolic tangent layer
    * \return Method of the backward hyperbolic tangent layer
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains input objects of the backward hyperbolic tangent layer
     * \return Structure that contains input objects of the backward hyperbolic tangent layer
     */
    virtual Input *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains parameters of the forward hyperbolic tangent layer
     * \return Structure that contains parameters of the forward hyperbolic tangent layer
     */
    virtual Parameter *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains results of the backward hyperbolic tangent layer
     * \return Structure that contains results of the backward hyperbolic tangent layer
     */
    services::SharedPtr<layers::backward::Result> getLayerResult() DAAL_C11_OVERRIDE
    {
        return _result;
    }

    /**
     * Returns the structure that contains the result of the backward hyperbolic tangent layer
     * \return Structure that contains the result of backward hyperbolic tangent layer
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of the backward hyperbolic tangent layer
     * \param[in] result  Structure to store  results of the backward hyperbolic tangent layer
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to a newly allocated backward hyperbolic tangent layer
     * with a copy of input objects of this backward hyperbolic tangent layer
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the backward hyperbolic tangent layer
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
} // namespace tanh
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
