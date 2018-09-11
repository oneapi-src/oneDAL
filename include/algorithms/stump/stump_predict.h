/* file: stump_predict.h */
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
//  Implementation of the interface for decision stump model-based prediction
//--
*/

#ifndef __STUMP_PREDICT_H__
#define __STUMP_PREDICT_H__

#include "algorithms/algorithm.h"
#include "algorithms/weak_learner/weak_learner_predict.h"
#include "algorithms/stump/stump_predict_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
/**
 * @defgroup stump_prediction_batch Batch
 * @ingroup stump_prediction
 * @{
 */
/**
 * \brief Contains classes to make prediction based on the decision stump model
 */
namespace prediction
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__STUMP__PREDICTION__METHOD"></a>
 * Available methods to make prediction based on the decision stump model
 */
enum Method
{
    defaultDense = 0        /*!< Default method */
};

/**
 * \brief Contains version 1.0 of Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__PREDICTION__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the decision stump prediction algorithm.
 *        It is associated with the daal::algorithms::stump::prediction::interface1::Batch class
 *        and supports methods to run based on the decision stump model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the decision stump prediction algorithm, double or float
 * \tparam method           Decision stump model-based prediction method, \ref Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public PredictionContainerIface
{
public:
    /**
     * Constructs a container for decision stump model-based prediction with a specified environment
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of decision stump model-based prediction
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__PREDICTION__BATCH"></a>
 * \brief Predicts results of the decision stump classification
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the decision stump prediction algorithm, double or float
 * \tparam method           Decision stump model-based prediction method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                                       Decision stump prediction methods
 *      - \ref classifier::prediction::NumericTableInputId  Identifiers of input Numeric Table objects
 *                                                          for the decision stump prediction algorithm
 *      - \ref classifier::prediction::ModelInputId         Identifiers of input %Model objects
 *                                                          for the decision stump prediction algorithm
 *      - \ref classifier::prediction::ResultId             Identifiers of the results of the decision stump prediction algorithm
 *
 * \par References
 *      - \ref interface1::Model "Model" class
 *      - \ref classifier::prediction::interface1::Input "classifier::prediction::Input" class
 *      - \ref classifier::prediction::interface1::Result "classifier::prediction::Result" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class Batch : public weak_learner::prediction::Batch
{
public:
    typedef weak_learner::prediction::Batch super;

    typedef algorithms::stump::prediction::Input InputType;
    typedef typename super::ParameterType        ParameterType;
    typedef typename super::ResultType           ResultType;

    InputType input;            /*!< %Input objects of the algorithm */

    Batch()
    {
        initialize();
    }

    /**
     * Constructs decision stump classification algorithm by copying input objects
     * of another decision stump classification algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : weak_learner::prediction::Batch(other), input(other.input)
    {
        initialize();
    }

    ~Batch() {}

    /**
     * Get input objects for the decision stump prediction algorithm
     * \return %Input objects for the decision stump prediction algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns a pointer to the newly allocated decision stump classification algorithm
     * with a copy of input objects and parameters of this decision stump classification algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    using weak_learner::prediction::Batch::_result;

    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, 0, 0);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _in = &input;
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _par = NULL;
    }
};
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace daal::algorithms::stump::prediction
/** @} */
}
}
} // namespace daal
#endif
