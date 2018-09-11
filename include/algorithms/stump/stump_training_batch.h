/* file: stump_training_batch.h */
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
//  Implementation of the interface for decision stump model-based training
//  in the batch processing mode
//--
*/

#ifndef __STUMP_TRAINING_BATCH_H__
#define __STUMP_TRAINING_BATCH_H__

#include "algorithms/weak_learner/weak_learner_training_batch.h"
#include "algorithms/stump/stump_training_types.h"

namespace daal
{
namespace algorithms
{
namespace stump
{
namespace training
{

namespace interface1
{
/**
 * @defgroup stump_training_batch Batch
 * @ingroup stump_training
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__TRAINING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the the decision stump training algorithm.
 *        It is associated with the daal::algorithms::stump::training::Batch class
 *        and supports methods to train the decision stump model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the the decision stump training algorithm, double or float
 * \tparam method           the decision stump training method, \ref Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public TrainingContainerIface<batch>
{
public:
    /**
     * Constructs a container for decision stump model-based training with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of decision stump model-based training in the batch processing mode
     */
    services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__STUMP__TRAINING__BATCH"></a>
 * \brief Trains the decision stump model
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the the decision stump training method, double or float
 * \tparam method           Decision stump training method, \ref Method
 *
 * \par Enumerations
 *      - \ref Method                         Decision stump training methods
 *      - \ref classifier::training::InputId  Identifiers of input objects for the decision stump training algorithm
 *      - \ref classifier::training::ResultId Identifiers of results of the decision stump training algorithm
 *
 * \par References
 *      - \ref classifier::training::interface1::Input "classifier::training::Input" class
 *      - \ref interface1::Model "Model" class
 *      - Result class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public weak_learner::training::Batch
{
public:
    typedef weak_learner::training::Batch super;

    typedef typename super::InputType           InputType;
    typedef typename super::ParameterType       ParameterType;
    typedef algorithms::stump::training::Result ResultType;

    ParameterType parameter;            /*!< \ref classifier::interface1::Parameter "Parameters" of the algorithm */
    InputType input;                    /*!< %Input data structure */

    Batch()
    {
        initialize();
    }

    /**
     * Constructs decision stump training algorithm by copying input objects
     * of another decision stump training algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : weak_learner::training::Batch(other),
        parameter(other.parameter), input(other.input)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Get input objects for the stump training algorithm
     * \return %Input objects for the stump training algorithm
     */
    InputType * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains computed results of the decision stump training algorithm
     * \return Structure that contains computed results of the decision stump training algorithm
     */
    training::ResultPtr getResult()
    {
        return ResultType::cast(_result);
    }

    /**
     * Resets the training results of the classification algorithm
     */
    services::Status resetResult() DAAL_C11_OVERRIDE
    {
        _result.reset(new ResultType());
        DAAL_CHECK(_result, services::ErrorNullResult)
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated decision stump training algorithm
     * with a copy of input objects and parameters of this decision stump training algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> * cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        training::ResultPtr res = getResult();
        DAAL_CHECK(res, services::ErrorNullResult);
        services::Status s = res->template allocate<algorithmFPType>(&input, _par, method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        _ac  = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _par = &parameter;
        _result.reset(new ResultType());
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace daal::algorithms::stump::training
}
}
} // namespace daal
#endif // __STUMP_TRAINING_BATCH_H__
