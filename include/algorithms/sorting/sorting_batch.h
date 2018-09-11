/* file: sorting_batch.h */
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
//  Implementation of the interface for the observation sorting algorithm
//  in the batch processing mode
//--
*/

#ifndef __SORTING_BATCH_H__
#define __SORTING_BATCH_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/sorting/sorting_types.h"

namespace daal
{
namespace algorithms
{
namespace sorting
{

namespace interface1
{
/**
 * @defgroup sorting_batch Batch
 * @ingroup sorting
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__SORTING__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the sorting algorithm.
 *        It is associated with the daal::algorithms::sorting::Batch class
 *        and supports methods of sorting computation in the batch processing mode
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the sorting algorithms, double or float
 * \tparam method           Sorting computation method, \ref daal::algorithms::sorting::Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the observation sorting algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    virtual ~BatchContainer();
    /**
     * Computes the result of the observation sorting algorithm in the batch processing mode
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__SORTING__BATCH"></a>
 * \brief Sorts the datasets by components of the random vector in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SORTING-ALGORITHM">Sorting algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the sorting, double or float
 * \tparam method           Sorting computation method, \ref daal::algorithms::sorting::Method
 *
 * \par Enumerations
 *      - \ref Method   Sorting computation methods
 *      - \ref InputId  Identifiers of sorting input objects
 *      - \ref ResultId Identifiers of sorting results
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::sorting::Input  InputType;
    typedef algorithms::sorting::Result ResultType;

    InputType input; /*!< %input data structure */

    /** Default constructor     */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs sorting algorithm by copying input objects and parameters
     * of another sorting algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : input(other.input)
    {
        initialize();
    }

    virtual ~Batch() {}

    /**
    * Returns method of the algorithm
    * \return Method of the algorithm
    */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int)method; }

    /**
     * Returns the structure that contains computed results of the sorting
     * \return Structure that contains computed results of the sorting
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of the sorting algorithms
     * \param[in] result Structure to store results of the sorting algorithms
     */
    services::Status setResult(const ResultPtr &result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated sorting algorithm
     * with a copy of input objects and parameters of this sorting algorithm
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<Batch<algorithmFPType, method> > clone() const
    {
        return services::SharedPtr<Batch<algorithmFPType, method> >(cloneImpl());
    }

protected:
    virtual Batch<algorithmFPType, method> *cloneImpl() const DAAL_C11_OVERRIDE
    {
        return new Batch<algorithmFPType, method>(*this);
    }

    virtual services::Status allocateResult() DAAL_C11_OVERRIDE
    {
        services::Status s = _result->allocate<algorithmFPType>(&input, method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
        _result.reset(new ResultType());
    }

    ResultPtr _result;

};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace daal::algorithms::sorting
} // namespace daal::algorithms
} // namespace daal
#endif
