/* file: saga_batch.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
//  Implementation of the interface for the Stochastic average gradient descent (SAGA) algorithm
//  in the batch processing mode
//--
*/

#ifndef __SAGA_BATCH_H__
#define __SAGA_BATCH_H__

#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/optimization_solver/iterative_solver/iterative_solver_batch.h"
#include "saga_types.h"

namespace daal
{
namespace algorithms
{
namespace optimization_solver
{
namespace saga
{
namespace interface1
{
/**
 * @defgroup saga_batch Batch
 * @ingroup saga
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SAGA__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the stochastic average gradient descent algorithm.
 *        This class is associated with daal::algorithms::optimization_solver::saga::BatchContainer class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Stochastic average gradient descent algorithm, double or float
 * \tparam method           Stochastic average gradient descent computation method, daal::algorithms::optimization_solver::saga::Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the Saga algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the Saga algorithm in the batch processing mode
     *
     * \return Status of computations
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__OPTIMIZATION_SOLVER__SAGA__BATCH"></a>
 * \brief Computes Stochastic average gradient descent in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SGD-ALGORITHM">Stochastic average gradient descent algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the Stochastic average gradient descent algorithm,
 *                          double or float
 * \tparam method           Stochastic average gradient descent computation method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for Stochastic average gradient descent
 *      - \ref iterative_solver::InputId  Identifiers of input objects for Stochastic average gradient descent
 *      - \ref iterative_solver::ResultId %Result identifiers for the Stochastic average gradient descent
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public iterative_solver::Batch
{
public:
    typedef algorithms::optimization_solver::saga::Input     InputType;
    typedef algorithms::optimization_solver::saga::Parameter ParameterType;
    typedef algorithms::optimization_solver::saga::Result    ResultType;

    InputType input;         /*!< %Input data structure */

    /** Default constructor */
    Batch(const sum_of_functions::BatchPtr& objectiveFunction = sum_of_functions::BatchPtr());

    /**
     * Constructs a Stochastic average gradient descent algorithm by copying input objects
     * of another Stochastic average gradient descent algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other);

    ~Batch()
    {
        delete _par;
    }
    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    ParameterType& parameter() { return *static_cast<ParameterType*>(_par); }

    /**
    * Gets parameter of the algorithm
    * \return parameter of the algorithm
    */
    const ParameterType& parameter() const { return *static_cast<const ParameterType*>(_par); }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Get input objects for the iterative solver algorithm
     * \return %Input objects for the iterative solver algorithm
     */
    virtual iterative_solver::Input * getInput() DAAL_C11_OVERRIDE { return &input; }

    /**
     * Get parameters of the iterative solver algorithm
     * \return Parameters of the iterative solver algorithm
     */
    virtual iterative_solver::Parameter * getParameter() DAAL_C11_OVERRIDE { return &parameter(); }

    /**
     * Creates user-allocated memory to store results of the iterative solver algorithm
     *
     * \return Status of computations
     */
    virtual services::Status createResult() DAAL_C11_OVERRIDE
    {
        _result = iterative_solver::ResultPtr(new ResultType());
        _res = NULL;
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated Stochastic average gradient descent algorithm with a copy of input objects
     * of this Stochastic average gradient descent algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

    /**
    *  Creates the instance of the class
    *  \return     New instance of the class
    */
    static services::SharedPtr<Batch<algorithmFPType, method> > create();

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = static_cast<ResultType*>(_result.get())->allocate<algorithmFPType>(&input, _par, (int)method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _result.reset(new ResultType());
    }
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace saga
} // namespace optimization_solver
} // namespace algorithm
} // namespace daal
#endif
