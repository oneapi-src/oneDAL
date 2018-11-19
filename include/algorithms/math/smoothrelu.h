/* file: smoothrelu.h */
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
//  Implementation of the interface for the SmoothReLU algorithm
//--
*/

#ifndef __SMOOTHRELU_H__
#define __SMOOTHRELU_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/math/smoothrelu_types.h"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace smoothrelu
{
namespace interface1
{
/**
 * @defgroup smoothrelu_batch Batch
 * @ingroup smoothrelu
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__SMOOTHRELU__BATCHCONTAINER"></a>
 * \brief Provides methods to run implementations of the SmoothReLU algorithm.
 *        This class is associated with daal::algorithms::math::smoothrelu::Batch class.
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SmoothReLU algorithm, double or float
 * \tparam method           SmoothReLU computation method, \ref daal::algorithms::math::smoothrelu::Method
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the SmoothReLU algorithm with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the SmoothReLU algorithm in the batch processing mode
     *
     * \return Status of computation
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__SMOOTHRELU__BATCH"></a>
 * \brief Computes SmoothReLU in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SMOOTHRELU-ALGORITHM">SmoothReLU algorithm description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the SmoothReLU algorithm,
 *                          double or float
 * \tparam method           SmoothReLU computation method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for SmoothReLU
 *      - \ref InputId  Identifiers of input objects for SmoothReLU
 *      - \ref ResultId %Result identifiers for the SmoothReLU
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::math::smoothrelu::Input  InputType;
    typedef algorithms::math::smoothrelu::Result ResultType;

    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs SmoothReLU algorithm by copying input objects of another SmoothReLU algorithm
     * \param[in] other An algorithm to be used as the source to initialize the input objects of the algorithm
     */
    Batch(const Batch<algorithmFPType, method> &other) : input(other.input)
    {
        initialize();
    }

    /**
     * Returns method of the algorithm
     * \return Method of the algorithm
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains results of SmoothReLU
     * \return Structure that contains results of SmoothReLU
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of SmoothReLU
     * \param[in] result  Structure to store  results of SmoothReLU
     *
     * \return Status of computation
     */
    services::Status setResult(const ResultPtr &result)
    {
        DAAL_CHECK(result, services::ErrorNullResult)
        _result = result;
        _res = _result.get();
        return services::Status();
    }

    InputType input;                 /*!< %Input data structure */

    /**
     * Returns a pointer to the newly allocated SmoothReLU algorithm
     * with a copy of input objects of this SmoothReLU algorithm
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
        _result.reset(new ResultType());
        services::Status s = _result->allocate<algorithmFPType>(&input, NULL, (int) method);
        _res = _result.get();
        return s;
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
    }
private:
    ResultPtr _result;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace smoothrelu
} // namespace math
} // namespace algorithms
} // namespace daal
#endif
