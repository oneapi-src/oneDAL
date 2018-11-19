/* file: logistic_cross_layer_backward.h */
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
//  Implementation of the interface for the backward logistic cross-entropy layer in the batch processing mode
//--
*/

#ifndef __NEURAL_NENTWORK_LOSS_LOGISTIC_CROSS_LAYER_BACKWARD_H__
#define __NEURAL_NENTWORK_LOSS_LOGISTIC_CROSS_LAYER_BACKWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/loss/loss_layer_backward.h"
#include "algorithms/neural_networks/layers/loss/logistic_cross_layer_types.h"
#include "algorithms/neural_networks/layers/loss/logistic_cross_layer_backward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace loss
{
namespace logistic_cross
{
namespace backward
{
namespace interface1
{
/**
 * @defgroup logistic_cross_backward_batch Batch
 * @ingroup logistic_cross_backward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOGISTIC_CROSS__BACKWARD__BATCHCONTAINER"></a>
* \brief Provides methods to run implementations of the of the backward logistic cross-entropy layer
*        This class is associated with the daal::algorithms::neural_networks::layers::loss::logistic_cross::backward::Batch class
*        and supports the method of backward logistic cross-entropy layer computation in the batch processing mode
*
* \tparam algorithmFPType  Data type to use in intermediate computations of backward logistic cross-entropy layer, double or float
* \tparam method           Computation method of the layer, \ref daal::algorithms::neural_networks::layers::loss::logistic_cross::Method
* \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
*/
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the backward logistic cross-entropy layer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the backward logistic cross-entropy layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LOSS__LOGISTIC_CROSS__BACKWARD__BATCH"></a>
 * \brief Provides methods for the backward logistic cross-entropy layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-DROPOUTBACKWARD-ALGORITHM">Backward logistic cross-entropy layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the backward logistic cross-entropy layer, double or float
 * \tparam method           Backward logistic cross-entropy layer method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                      Computation methods for the backward logistic cross-entropy layer
 *      - \ref backward::InputId           Identifiers of input objects for the backward logistic cross-entropy layer
 *      - \ref LayerDataId                 Identifiers of collection in input objects for the backward logistic cross-entropy layer
 *      - \ref backward::InputLayerDataId  Identifiers of extra results computed by the forward logistic cross-entropy layer
 *      - \ref backward::ResultId          Identifiers of result objects for the backward logistic cross-entropy layer
 *
 * \par References
 *      - \ref forward::interface1::Batch "forward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public loss::backward::Batch
{
public:
    typedef loss::backward::Batch super;

    typedef algorithms::neural_networks::layers::loss::logistic_cross::backward::Input     InputType;
    typedef algorithms::neural_networks::layers::loss::logistic_cross::Parameter           ParameterType;
    typedef algorithms::neural_networks::layers::loss::logistic_cross::backward::Result    ResultType;

    ParameterType &parameter; /*!< Backward logistic cross-entropy layer \ref interface1::Parameter "parameters" */
    InputType input;          /*!< Backward logistic cross-entropy layer input */

    /** Default constructor */
    Batch() : parameter(_defaultParameter)
    {
        initialize();
    };

    /**
     * Constructs a backward logistic cross-entropy layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(Parameter& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }


    /**
     * Constructs a backward logistic cross-entropy layer by copying input objects
     * and parameters of another backward logistic cross-entropy layer in the batch processing mode
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
     * Returns the structure that contains the input objects of backward logistic cross-entropy layer
     * \return Structure that contains the input objects of backward logistic cross-entropy layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains the parameters of the backward logistic cross-entropy layer
     * \return Structure that contains the parameters of the backward logistic cross-entropy layer
     */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains result of the backward logistic cross-entropy layer
     * \return Structure that contains result of the backward logistic cross-entropy layer
     */
    layers::backward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return _result;
    }

    /**
     * Returns the structure that contains the result of the backward logistic cross-entropy layer
     * \return Structure that contains the result of the backward logistic cross-entropy layer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the result of the backward logistic cross-entropy layer
     * \param[in] result Structure to store the result of the backward logistic cross-entropy layer
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
     * Returns a pointer to a newly allocated backward logistic cross-entropy layer
     * with a copy of the input objects and parameters for this backward logistic cross-entropy layer
     * in the batch processing mode
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    * Allocates memory to store the result of the backward logistic cross-entropy layer
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
} // namespace logistic_cross
} // namespace loss
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal
#endif
