/* file: prelu_layer_forward.h */
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
//  Implementation of the interface for the forward parametric rectified linear unit (pReLU) layer
//  in the batch processing mode
//--
*/

#ifndef __PRELU_LAYER_FORWARD_H__
#define __PRELU_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/prelu/prelu_layer_types.h"
#include "algorithms/neural_networks/layers/prelu/prelu_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the preluu layer
 */
namespace prelu
{
namespace forward
{
namespace interface1
{
/**
 * @defgroup prelu_forward_batch Batch
 * @ingroup prelu_forward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__FORWARD__BATCHCONTAINER"></a>
* \brief Provides methods to run implementations of the of the forward prelu layer
*        This class is associated with the daal::algorithms::neural_networks::layers::prelu::forward::Batch class
*        and supports the method of forward prelu layer computation in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations of forward prelu layer, double or float
* \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::prelu::Method
* \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
*/
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public layers::forward::LayerContainerIfaceImpl
{
public:
    /**
    * Constructs a container for the forward pReLU layer with a specified environment
    * in the batch processing mode
    * \param[in] daalEnv   Environment object
    */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the forward pReLU layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    services::Status setupCompute() DAAL_C11_OVERRIDE;
    virtual services::Status allocateInput() DAAL_C11_OVERRIDE
    {
        Input *input = static_cast<Input *>(_in);
        return input->allocate<algorithmFPType>(_par, (int) method);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__FORWARD__BATCH"></a>
 * \brief Computes the results of the forward prelu layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-PRELUFORWARD-ALGORITHM">Forward prelu layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the forward prelu layer, double or float
 * \tparam method           The forward prelu layer computation method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                     Computation methods for the forward prelu layer
 *      - \ref forward::InputId           Identifiers of input objects for the forward prelu layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward prelu layer
 *      - \ref forward::ResultLayerDataId Identifiers of extra results computed by the forward prelu layer
 *      - \ref LayerDataId                Identifiers of collection in result objects for the forward prelu layer
 *
 * \par References
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public layers::forward::LayerIfaceImpl
{
public:
    typedef layers::forward::LayerIfaceImpl super;

    typedef algorithms::neural_networks::layers::prelu::forward::Input     InputType;
    typedef algorithms::neural_networks::layers::prelu::Parameter          ParameterType;
    typedef algorithms::neural_networks::layers::prelu::forward::Result    ResultType;

    ParameterType &parameter;  /*!< \ref interface1::Parameter "Parameters" of the layer */
    InputType input;           /*!< %Input objects of the layer */

    /** \brief Default constructor */
    Batch() : parameter(_defaultParameter)
    {
        initialize();
    };

    /**
     * Constructs a forward prelu layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(ParameterType& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs a forward prelu layer by copying input objects of
     * another forward prelu layer
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : super(other),
        _defaultParameter(other.parameter), parameter(_defaultParameter), input(other.input)
    {
        initialize();
    }

    /**
    * Returns method of the forward prelu layer
    * \return Method of the forward prelu layer
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains input objects of the forward prelu layer
     * \return Structure that contains input objects of the forward prelu layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
    * Returns the structure that contains parameters of the forward prelu layer
    * \return Structure that contains parameters of the forward prelu layer
    */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains the result of the forward prelu layer
     * \return Structure that contains the result of the forward prelu layer
     */
    layers::forward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return _result;
    }

    /**
     * Returns the structure that contains the result of the forward prelu layer
     * \return Structure that contains the result of forward prelu layer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of the forward prelu layer
     * \param[in] result  Structure to store  results of the forward prelu layer
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
     * Returns a pointer to the newly allocated forward prelu layer
     * with a copy of input objects of this forward prelu layer
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the forward prelu layer
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
} // namespace prelu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
