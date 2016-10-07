/* file: lrn_layer_forward.h */
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
//  Implementation of the interface for the forward local response normalization layer
//  in the batch processing mode
//--
*/

#ifndef __LRN_LAYER_FORWARD_H__
#define __LRN_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/lrn/lrn_layer_types.h"
#include "algorithms/neural_networks/layers/lrn/lrn_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace lrn
{
namespace forward
{
namespace interface1
{
/**
 * @defgroup lrn_forward_batch Batch
 * @ingroup lrn_forward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__FORWARD__BATCHCONTAINER"></a>
* \brief Provides methods to run implementations of the of the forward local response normalization layer
*        This class is associated with the daal::algorithms::neural_networks::layers::lrn::forward::Batch class
*        and supports the method of forward local response normalization layer computation in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations of forward local response normalization layer, double or float
* \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::lrn::Method
* \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
*/
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the forward local response normalization layer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the forward local response normalization layer in the batch processing mode
     */
    void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LRN__FORWARD__BATCH"></a>
 * \brief Provides methods for the forward local response normalization layer in the batch processing mode
 * \n<a href="DAAL-REF-LRNFORWARD-ALGORITHM">Forward local response normalization layer description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the forward local response normalization layer, double or float
 * \tparam method           Forward local response normalization layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                     Computation methods for the forward local response normalization layer
 *      - \ref forward::InputId           Identifiers of input objects for the forward local response normalization layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward local response normalization layer
 *      - \ref forward::ResultLayerDataId Identifiers of extra results computed by the forward local response normalization layer
 *      - \ref LayerDataId                Identifiers of collection in result objects for the forward local response normalization layer
 *
 * \par References
 *      - \ref interface1::Parameter "Parameter" class
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = float, Method method = defaultDense>
class Batch : public layers::forward::LayerIface
{
public:
    Parameter &parameter; /*!< Forward local response normalization layer parameters */
    Input input;          /*!< Forward local response normalization layer input */

    /** Default constructor */
    Batch() : parameter(_defaultParameter)
    {
        initialize();
    };


    /**
     * Constructs a forward local response normalization layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(Parameter& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }


    /**
     * Constructs a forward local response normalization layer by copying input objects
     * and parameters of another forward local response normalization layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the layer
     */
    Batch(const Batch<algorithmFPType, method> &other) : _defaultParameter(other.parameter), parameter(_defaultParameter)
    {
        initialize();
        input.set(layers::forward::data, other.input.get(layers::forward::data));
        input.set(layers::forward::weights, other.input.get(layers::forward::weights));
        input.set(layers::forward::biases, other.input.get(layers::forward::biases));
    }

    /**
    * Returns the method of the layer
    * \return Method of the layer
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the input objects of the forward local response normalization layer
     * \return Structure that contains the input objects of the forward local response normalization layer
     */
    virtual Input *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains the parameters of the forward local response normalization layer
     * \return Structure that contains the parameters of the forward local response normalization layer
     */
    virtual Parameter *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains result of the forward local response normalization layer
     * \return Structure that contains result of the forward local response normalization layer
     */
    services::SharedPtr<layers::forward::Result> getLayerResult() DAAL_C11_OVERRIDE
    {
        return getResult();
    }

    /**
     * Returns the structure that contains the result of the forward local response normalization layer
     * \return Structure that contains the result of the forward local response normalization layer
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward local response normalization layer
     * \param[in] result Structure to store the result of the forward local response normalization layer
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    /**
     * Returns a pointer to a newly allocated forward local response normalization layer
     * with a copy of the input objects and parameters for this forward local response normalization layer
     * in the batch processing mode
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the forward local response normalization layer
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

    void setParameter() DAAL_C11_OVERRIDE { _par = &parameter; };

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
} // namespace forward
} // namespace lrn
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
