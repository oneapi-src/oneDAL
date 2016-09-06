/* file: relu.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/*
//++
//  Implementation of the interface for the rectified linear function
//--
*/

#ifndef __RELU_H__
#define __RELU_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "services/daal_defines.h"
#include "algorithms/math/relu_types.h"

namespace daal
{
namespace algorithms
{
namespace math
{
namespace relu
{
namespace interface1
{
/** @defgroup relu_batch Batch
 * @ingroup relu
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__RELU__BATCHCONTAINER"></a>
 * \brief Class containing methods for the rectified linear function computing using algorithmFPType precision arithmetic
 */
template<typename algorithmFPType, Method method, CpuType cpu>
class DAAL_EXPORT BatchContainer : public daal::algorithms::AnalysisContainerIface<batch>
{
public:
    /**
     * Constructs a container for the rectified linear function with a specified environment
     * in the batch processing mode
     * \param[in] daalEnv   Environment object
     */
    BatchContainer(daal::services::Environment::env *daalEnv);
    /** Default destructor */
    ~BatchContainer();
    /**
     * Computes the result of the rectified linear function in the batch processing mode
     */
    virtual void compute() DAAL_C11_OVERRIDE;
};

/**
 * <a name="DAAL-CLASS-ALGORITHMS__MATH__RELU__BATCH"></a>
 * \brief Computes the rectified linear function in the batch processing mode.
 * \n<a href="DAAL-REF-RELU-ALGORITHM">rectified linear function description and usage models</a>
 *
 * \tparam algorithmFPType  Data type to use in intermediate computations for the rectified linear function,
 *                          double or float
 * \tparam method           the rectified linear function computation method
 *
 * \par Enumerations
 *      - \ref Method   Computation methods for the rectified linear function
 *      - \ref InputId  Identifiers of input objects for the rectified linear function
 *      - \ref ResultId Result identifiers for the rectified linear function
 *
 * \par References
 *      - \ref interface1::Input "Input" class
 *      - Result class
 */
template<typename algorithmFPType = double, Method method = defaultDense>
class DAAL_EXPORT Batch : public daal::algorithms::Analysis<batch>
{
public:
    /** Default constructor */
    Batch()
    {
        initialize();
    }

    /**
     * Constructs the rectified linear function by copying input objects of another rectified linear function
     * \param[in] other function to be used as the source to initialize the input objects of the rectified linear function
     */
    Batch(const Batch<algorithmFPType, method> &other)
    {
        initialize();
        input.set(data, other.input.get(data));
    }

    /**
     * Returns the method of the rectified linear function
     * \return Method of the function
     */
    virtual int getMethod() const DAAL_C11_OVERRIDE { return(int) method; }

    /**
     * Returns the structure that contains the result of the rectified linear function
     * \return Structure that contains results of the rectified linear function
     */
    services::SharedPtr<Result> getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store the result of the rectified linear function
     * \param[in] result  Structure to store the result of the rectified linear function
     */
    void setResult(const services::SharedPtr<Result>& result)
    {
        DAAL_CHECK(result, ErrorNullResult)
        _result = result;
        _res = _result.get();
    }

    Input input;                 /*!< %Input data structure */

    /**
     * Returns a pointer to a newly allocated rectified linear function
     * with a copy of input objects of this rectified linear function
     * \return Pointer to the newly allocated rectified linear function
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

    virtual void allocateResult() DAAL_C11_OVERRIDE
    {
        _result = services::SharedPtr<Result>(new Result());
        _result->allocate<algorithmFPType>(&input, NULL, (int) method);
        _res = _result.get();
    }

    void initialize()
    {
        Analysis<batch>::_ac = new __DAAL_ALGORITHM_CONTAINER(batch, BatchContainer, algorithmFPType, method)(&_env);
        _in  = &input;
    }
private:
    services::SharedPtr<Result> _result;
};
/** @} */
} // namespace interface1
using interface1::BatchContainer;
using interface1::Batch;

} // namespace relu
} // namespace math
} // namespace algorithms
} // namespace daal
#endif
