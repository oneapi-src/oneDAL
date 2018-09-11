/* file: batch_normalization_layer_forward.h */
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
//  Implementation of the interface for the forward batch normalization layer
//  in the batch processing mode
//--
*/

#ifndef __BATCH_NORMALIZATION_LAYER_FORWARD_H__
#define __BATCH_NORMALIZATION_LAYER_FORWARD_H__

#include "algorithms/algorithm.h"
#include "data_management/data/tensor.h"
#include "services/daal_defines.h"
#include "algorithms/neural_networks/layers/layer.h"
#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_types.h"
#include "algorithms/neural_networks/layers/batch_normalization/batch_normalization_layer_forward_types.h"

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace layers
{
namespace batch_normalization
{
namespace forward
{
/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface
 */
namespace interface1
{
/**
 * @defgroup batch_normalization_forward_batch Batch
 * @ingroup batch_normalization_forward
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the forward batch normalization layer.
 *        This class is associated with the \ref forward::interface1::Batch "forward::Batch" class
 *        and supports the method of forward batch normalization layer computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of forward batch normalization layer, double or float
 * \tparam method           Computation method of the layer, batch_normalization::Method
 * \tparam cpu              Version of the cpu-specific implementation of the layer, \ref daal::CpuType
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public layers::forward::LayerContainerIfaceImpl
{
public:
    /**
     * Constructs a container for the forward batch normalization layer with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the forward batch normalization layer in the batch processing mode
     *
     * \return Status of computations
     */
    services::Status compute() DAAL_C11_OVERRIDE;
    services::Status setupCompute() DAAL_C11_OVERRIDE;
    services::Status resetCompute() DAAL_C11_OVERRIDE;
    virtual services::Status allocateInput() DAAL_C11_OVERRIDE
    {
        Input *input = static_cast<Input *>(_in);
        return input->allocate<algorithmFPType>(_par, (int) method);
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__BATCH_NORMALIZATION__FORWARD__BATCH"></a>
 * \brief Provides methods for the forward batch normalization layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-BATCH_NORMALIZATIONFORWARD-ALGORITHM">Forward batch normalization layer description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations of the forward batch normalization layer, double or float
 * \tparam method           Forward batch normalization layer method, batch_normalization::Method
 *
 * \par Enumerations
 *      - \ref Method                     Computation methods for the forward batch normalization layer
 *      - \ref forward::InputId           Identifiers of input objects for the forward batch normalization layer
 *      - \ref forward::ResultId          Identifiers of result objects for the forward batch normalization layer
 *      - \ref forward::ResultLayerDataId Identifiers of extra results computed by the forward batch normalization layer
 *      - \ref LayerDataId                Identifiers of collection in result objects for the forward batch normalizationv layer
 *
 * \par References
 *      - \ref backward::interface1::Batch "backward::Batch" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public layers::forward::LayerIfaceImpl
{
public:
    typedef layers::forward::LayerIfaceImpl super;

    typedef algorithms::neural_networks::layers::batch_normalization::forward::Input     InputType;
    typedef algorithms::neural_networks::layers::batch_normalization::Parameter          ParameterType;
    typedef algorithms::neural_networks::layers::batch_normalization::forward::Result    ResultType;

    ParameterType &parameter; /*!< Forward batch normalization layer \ref interface1::Parameter "parameters" */
    InputType input;          /*!< Forward batch normalization layer input */

    /** Default constructor */
    Batch() : parameter(_defaultParameter)
    {
        initialize();
    }

    /**
     * Constructs a forward batch normalization layer in the batch processing mode
     * and initializes its parameter with the provided parameter
     * \param[in] parameter Parameter to initialize the parameter of the layer
     */
    Batch(ParameterType& parameter) : parameter(parameter), _defaultParameter(parameter)
    {
        initialize();
    }

    /**
     * Constructs a forward batch normalization layer by copying input objects
     * and parameters of another forward batch normalization layer in the batch processing mode
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
     * Returns the structure that contains the input objects of the forward batch normalization layer
     * \return Structure that contains the input objects of the forward batch normalization layer
     */
    virtual InputType *getLayerInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns the structure that contains the parameters of the forward batch normalization layer
     * \return Structure that contains the parameters of the forward batch normalization layer
     */
    virtual ParameterType *getLayerParameter() DAAL_C11_OVERRIDE { return &parameter; };

    /**
     * Returns the structure that contains result of the forward batch normalization layer
     * \return Structure that contains result of the forward batch normalization layer
     */
    layers::forward::ResultPtr getLayerResult() DAAL_C11_OVERRIDE
    {
        return getResult();
    }

    /**
     * Returns the structure that contains the result of the forward batch normalization layer
     * \return Structure that contains the result of the forward batch normalization layer
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward batch normalization layer
     * \param[in] result Structure to store the result of the forward batch normalization layer
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
     * Returns a pointer to a newly allocated forward batch normalization layer
     * with a copy of the input objects and parameters for this forward batch normalization layer
     * in the batch processing mode
     * \return Pointer to the newly allocated layer
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory to store the result of the forward batch normalization layer
     *
     * \return Status of computations
     */
    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = this->_result->template allocate<algorithmFPType>(&(this->input), &parameter, (int) method);
        this->_res = this->_result.get();
        return s;
    }

    /**
     * Returns the layer that corresponds to this layer on the prediction stage
     * \return The layer that corresponds to this layer on the prediction stage
     */
    virtual layers::forward::LayerIfacePtr getLayerForPrediction() const DAAL_C11_OVERRIDE
    {
        services::SharedPtr<Batch<algorithmFPType, method> > seflCopy = clone();
        if (_result)
        {
            seflCopy->input.set(forward::populationMean, _result->get(auxPopulationMean));
            seflCopy->input.set(forward::populationVariance, _result->get(auxPopulationVariance));
        }
        return services::staticPointerCast<layers::forward::LayerIface>(seflCopy);
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
} // namespace batch_normalization
} // namespace layers
} // namespace neural_networks
} // namespace algorithms
} // namespace daal

#endif
