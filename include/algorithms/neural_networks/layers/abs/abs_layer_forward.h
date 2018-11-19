/* file: abs_layer_forward.h */
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
//  Implementation of the interface for the forward absolute value (abs) layer
//  in the batch processing mode
//--
*/

#ifndef __ABS_LAYER_FORWARD_H__
#define __ABS_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/abs/abs_layer_types.h"
#include "algorithms/neural_networks/layers/abs/abs_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace abs
{
namespace forward
{
namespace interface1
{
/**
 * @defgroup abs_layers_forward_batch Batch
 * @ingroup abs_layers_forward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ABS__FORWARD__BATCHCONTAINER"></a>
* \brief Provides methods to run implementations of the of the forward abs layer
*        This class is associated with the daal::algorithms::neural_networks::layers::abs::forward::Batch class
*        and supports the method of forward abs layer computation in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations of forward abs layer, double or float
* \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::abs::Method
* \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
*/
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public layers::forward::LayerContainerIfaceImpl
{
public:
    /**
    * Constructs a container for the forward abs layer with a specified environment
    * \param[in] daalEnv   Environment object
    */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the forward abs layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__ABS__FORWARD__BATCH"></a>
 * \brief Computes the result of the forward abs layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-ABSFORWARD-ALGORITHM">Forward abs layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the forward abs layer, double or float
 * \tparam method           Forward abs layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                     Computation methods for the forward abs layer
 *      - \ref forward::InputId           Identifiers of input objects for the forward abs layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward abs layer
 *      - \ref LayerDataId                Identifiers of extra results computed by the forward abs layer
 *
 * \par References
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public layers::forward::LayerIfaceImpl
{
public:
    typedef layers::forward::LayerIfaceImpl super;

    typedef algorithms::neural_networks::layers::abs::forward::Input     InputType;
    typedef algorithms::neural_networks::layers::abs::Parameter          ParameterType;
    typedef algorithms::neural_networks::layers::abs::forward::Result    ResultType;

    ParameterType &parameter;  /*!< Abs layer \ref interface1::Parameter "parameters" structure */
    InputType input;          /*!< %Input objects of the layer */

    /** Default constructor */
    Batch() : parameter(_defaultParameter)
    {
        initialize();
    };

    /**
     * Constructs a forward abs layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(ParameterType& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs a forward abs layer by copying input objects
     * and parameters of another forward abs layer in the batch processing mode
     * \param[in] other Algorithm to use as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : super(other),
        _defaultParameter(other.parameter), parameter(_defaultParameter), input(other.input)
    {
        initialize();
    }

    /**
    * Returns the method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains input objects of the abs forward layer
     * \return Structure that contains input objects of the abs forward layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains parameters of the forward abs layer
     * \return Structure that contains parameters of the forward abs layer
     */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains result of the forward abs layer
     * \return Structure that contains result of the forward abs layer
     */
    layers::forward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return getResult();
    }

    /**
     * Returns the structure that contains result of the forward abs layer
     * \return Structure that contains result of the forward abs layer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store result of the forward abs layer
     * \param[in] result  Structure to store result of the forward abs layer
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
     * Returns a pointer to the newly allocated forward abs layer
     * with a copy of input objects and parameters of this forward abs layer
     * in the batch processing mode
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the forward abs layer
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
} // namespace abs
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
