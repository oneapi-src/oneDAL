/* file: relu_layer_backward.h */
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
//  Implementation of the interface for the backward rectified linear unit (ReLU) layer
//  in the batch processing mode
//--
*/

#ifndef __RELU_LAYER_BACKWARD_H__
#define __RELU_LAYER_BACKWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/relu/relu_layer_types.h"
#include "algorithms/neural_networks/layers/relu/relu_layer_backward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
/**
 * \brief Contains classes for the relu layer
 */
namespace relu
{
/**
 * \brief Contains classes for the backward relu layer
 */
namespace backward
{
namespace interface1
{
/**
 * @defgroup relu_layers_backward_batch Batch
 * @ingroup relu_layers_backward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RELU__BACKWARD__BATCHCONTAINER"></a>
* \brief Provides methods to run implementations of the of the backward relu layer
*        This class is associated with the daal::algorithms::neural_networks::layers::relu::backward::Batch class
*        and supports the method of backward relu layer computation in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations of backward relu layer, double or float
* \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::relu::Method
* \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
*/
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
    * Constructs a container for the backward ReLU layer with a specified environment
    * in the batch processing mode
    * \param[in] daalEnv   Environment object
    */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the backward ReLU layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__RELU__BACKWARD__BATCH"></a>
 * \brief Computes the results of the backward relu layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-RELUBACKWARD-ALGORITHM">Backward relu layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the backward relu layer, double or float
 * \tparam method           The backward relu layer computation method, \ref Method
 * \par Enumerations
 *      - \ref Method                      Computation methods for the backward relu layer
 *      - \ref backward::InputId           Identifiers of input objects for the backward relu layer
 *      - \ref LayerDataId                 Identifiers of collection in input objects for the relu layer
 *      - \ref backward::InputLayerDataId  Identifiers of extra results computed by the backward relu layer
 *      - \ref backward::ResultId          Identifiers of result objects for the backward relu layer
 *
 * \par References
 *      - \ref forward::interface1::Batch "forward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public layers::backward::LayerIfaceImpl
{
public:
    typedef layers::backward::LayerIfaceImpl super;

    typedef algorithms::neural_networks::layers::relu::backward::Input     InputType;
    typedef algorithms::neural_networks::layers::relu::Parameter           ParameterType;
    typedef algorithms::neural_networks::layers::relu::backward::Result    ResultType;

    ParameterType &parameter; /*!< relu layer \ref interface1::Parameter "parameters" structure */
    InputType input;         /*!< %Input objects of the layer */

    /** Default constructor */
    Batch() : parameter(_defaultParameter)
    {
        initialize();
    }

    /**
     * Constructs a backward relu layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(Parameter& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs the backward relu layer by copying input objects of
     * another backward relu layer in the batch processing mode
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  of the backward relu layer
     */
    Batch(const Batch<algorithmFPType, method> &other) : super(other),
        _defaultParameter(other.parameter), parameter(_defaultParameter), input(other.input)
    {
        initialize();
    }

    /**
    * Returns method of the backward relu layer
    * \return Method of the backward relu layer
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains input objects of the backward relu layer
     * \return Structure that contains input objects of the backward relu layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains parameters of the backward relu layer
     * \return Structure that contains parameters of the backward relu layer
     */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains results of the backward relu layer
     * \return Structure that contains results of the backward relu layer
     */
    layers::backward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return _result;
    }

    /**
     * Returns the structure that contains the result of the backward relu layer
     * \return Structure that contains the result of backward relu layer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of the backward relu layer
     * \param[in] result  Structure to store  results of the backward relu layer
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
     * Returns a pointer to a newly allocated backward relu layer
     * with a copy of input objects of this backward relu layer
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the backward relu layer
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
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
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
} // namespace backward
} // namespace relu
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
