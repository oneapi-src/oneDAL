/* file: kernel_function.h */
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
//  Implementation of the kernel function interface.
//--
*/

#ifndef __KERNEL_FUNCTION_H__
#define __KERNEL_FUNCTION_H__

#include "algorithms/algorithm.h"
#include "data_management/data/numeric_table.h"
#include "algorithms/kernel_function/kernel_function_types.h"

namespace daal
{
namespace algorithms
{
namespace kernel_function
{

namespace interface1
{
/**
 * @addtogroup kernel_function
 * @{
 */
/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__KERNELIFACE"></a>
 * \brief Abstract class that specifies the interface of the algorithms
 *        for computing kernel functions in the batch processing mode
 */
class KernelIface : public daal::algorithms::Analysis<batch>
{
public:
    typedef algorithms::kernel_function::Input         InputType;
    typedef algorithms::kernel_function::ParameterBase ParameterType;
    typedef algorithms::kernel_function::Result        ResultType;

    KernelIface()
    {
        initialize();
    }

    /**
     * Constructs an algorithm for computing kernel functions by copying input objects and parameters
     * of another algorithm for computing kernel functions
     * \param[in] other An algorithm to be used as the source to initialize the input objects
     *                  and parameters of the algorithm
     */
    KernelIface(const KernelIface &other)
    {
        initialize();
    }

    /**
     * Get input objects for the kernel function algorithm
     * \return %Input objects for the kernel function algorithm
     */
    virtual Input * getInput() = 0;

    /**
     * Get parameters of the kernel function algorithm
     * \return Parameters of the kernel function algorithm
     */
    virtual ParameterBase * getParameter() = 0;

    virtual ~KernelIface() {}

    /**
     * Returns the structure that contains computed results of the kernel function algorithm
     * \returns the Structure that contains computed results of the kernel function algorithm
     */
    ResultPtr getResult()
    {
        return _result;
    }

    /**
     * Registers user-allocated memory to store results of the kernel function algorithm
     * \param[in] res  Structure to store the results
     */
    services::Status setResult(const ResultPtr& res)
    {
        DAAL_CHECK(res, services::ErrorNullResult)
        _result = res;
        _res = _result.get();
        return services::Status();
    }

    /**
     * Returns a pointer to the newly allocated algorithm for computing kernel functions with a copy of input objects
     * and parameters of this algorithm for computing kernel functions
     * \return Pointer to the newly allocated algorithm
     */
    services::SharedPtr<KernelIface> clone() const
    {
        return services::SharedPtr<KernelIface>(cloneImpl());
    }

protected:
    void initialize()
    {
        _result = ResultPtr(new kernel_function::Result());
    }
    virtual KernelIface * cloneImpl() const DAAL_C11_OVERRIDE = 0;
    ResultPtr _result;
};
typedef services::SharedPtr<KernelIface> KernelIfacePtr;
/** @} */
} // namespace interface1
using interface1::KernelIface;
using interface1::KernelIfacePtr;

} // namespace kernel_function
} // namespace algorithm
} // namespace daal
#endif
