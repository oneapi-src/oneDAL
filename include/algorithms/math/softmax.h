/* file: softmax.h */
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
//  Implementation of the interface for the softmax function
//--
*/

#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/math/softmax_types.h"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace softmax
{
namespace interface1
{
/**
 * @defgroup softmax_batch Batch
 * @ingroup softmax
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__SOFTMAX__BATCHCONTAINER"></a>
 * \brief Class containing methods for the softmax function computing using algorithmFPType precision arithmetic
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the softmax function with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the softmax function in the batch processing mode
     *
     * \return Status of computation
     */
    virtual services::Status compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__SOFTMAX__BATCH"></a>
 * \brief Computes the softmax function in the batch processing mode.
 * <!-- \n<a href="DAAL-REF-SOFTMAX-ALGORITHM">softmax function description and usage models</a> -->
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the softmax function,
 *                          double or float
 * \tparam method           the softmax function computation method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the softmax function
 *      - \ref InputId  Identifiers of input objects for the softmax function
 *      - \ref ResultId %Result identifiers for the softmax function
 */
template<typename algorithmFPType = DAAL_ALGORITHM_FP_TYPE, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::math::softmax::Input  InputType;
    typedef algorithms::math::softmax::Result ResultType;

    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs the softmax function by copying input objects of another softmax function
     * \param[in] other function to be used as the source to initialize the input objects of the softmax function
     */
    Batch(const Batch<algorithmFPType, method> &other) : input(other.input)
    {
        initialize();
    }

    /**
     * Returns method of the softmax function
     * \return Method of the softmax function
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the result of the softmax function
     * \return Structure that contains the result of the softmax function
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the result of the softmax function
     * \param[in] result  Structure to store the result of the softmax function
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
     * Returns a pointer to a newly allocated softmax function
     * with a copy of input objects of this softmax function
     * \return Pointer to the newly allocated softmax function
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

} // namespace softmax
} // namespace math
} // namespace algorithms
} // namespace daal
#endif
