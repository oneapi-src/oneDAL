/* file: precomputed_batch.h */
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
//  Implementation of the of the objective function with precomputed characteristics.
//--
*/

#ifndef __PRECOMPUTED_BATCH_H__
#define __PRECOMPUTED_BATCH_H__

#include "algorithms/algorithm.h"
#include "services/daal_defines.h"
#include "objective_function_batch.h"
#include "sum_of_functions_batch.h"
#include "precomputed_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace precomputed
{
namespace interface1
{
/**
 * @defgroup precomputed_batch Batch
 * @ingroup precomputed
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__PRECOMPUTED__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the objective function with precomputed characteristics.
 *        This class is associated with the Batch class and supports the method of computing
 *        the objective function with precomputed characteristics in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the objective function with precomputed characteristics, double or float
 * \tparam method           The objective function with precomputed characteristics method
 */

template<typename algorithmFPType, Method method>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the objective function with precomputed characteristics with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv) {}
    virtual ~BatchContainer() {}
    /**
     * Runs implementations of the objective function with precomputed characteristics in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute()
    {
        /* empty compute */
        return services::Status();
    }
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__PRECOMPUTED__BATCH"></a>
 * \brief Computes the objective function with precomputed characteristics in the batch processing mode.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the objective function with precomputed characteristics, double or float
 * \tparam method           The objective function with precomputed characteristics method
 *
 * \par Enumerations
 *      - ResultId Result identifiers for the precomputed objective function
 *
 * \par References
 * <!--     - <a href="DAAL-REF-PRECOMPUTED-ALGORITHM">The objective function with precomputed characteristics algorithm description and usage models</a> -->
 *      - \ref objective_function::interface1::Result "objective_function::Result" class
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public sum_of_functions::Batch
{
public:
    typedef sum_of_functions::Batch super;

    typedef typename super::InputType     InputType;
    typedef typename super::ParameterType ParameterType;
    typedef typename super::ResultType    ResultType;

    InputType input;           /*!< %Input data structure
                                                  \note The algorithm does not use the provided input objects */
    ParameterType parameter;   /*!< %Parameter data structure
                                                  \note The algorithm does not use the provided parameters */
    /**
     *  Main constructor
     */
    Batch() : parameter(1), super(1, &input, &parameter)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
     * Constructs the objective function with precomputed characteristics algorithm by copying input objects and parameters
     * of another objective function with precomputed characteristics algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) :
        parameter(other.parameter), super(other.parameter.numberOfTerms, &input, &parameter), input(other.input)
    {
        initialize();
        const ResultType *otherResult = const_cast<Batch<algorithmFPType, method> &>(other).getResult().get();
        if (otherResult)
        {
            bool isResultInitialized = false;
            isResultInitialized = (isResultInitialized || otherResult->get(objective_function::gradientIdx));
            _result->set(objective_function::gradientIdx, otherResult->get(objective_function::gradientIdx));
            isResultInitialized = (isResultInitialized || otherResult->get(objective_function::valueIdx));
            _result->set(objective_function::valueIdx,    otherResult->get(objective_function::valueIdx));
            isResultInitialized = (isResultInitialized || otherResult->get(objective_function::hessianIdx));
            _result->set(objective_function::hessianIdx,  otherResult->get(objective_function::hessianIdx));
            if (isResultInitialized)
            {
                _res = _result.get();
            }
        }
    }

    /**
     * Returns the method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return (int)method; }

    /**
     * Returns a pointer to the newly allocated objective function with precomputed characteristics algorithm
     * with a copy of input objects of this objective function with precomputed characteristics algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
     * Allocates memory buffers needed for the computations
     *
     * \return Status of computations
     */
    services::Status allocate()
    {
        return allocateResult();
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, &parameter, (int) method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new BatchContainer<algorithmFPType, method>(&_env);
        _in  = &input;
        _par = &parameter;
        _result = objective_function::ResultPtr(new ResultType());
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace precomputed
} // namespace optimization_solver
} // namespace algorithms
} // namespace daal
#endif
