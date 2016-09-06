/* file: kernel_function.h */
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
    ParameterBase *parameterBase;
    Input *inputBase;

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

    virtual ~KernelIface() {}
    virtual services::SharedPtr<Result> getResult() = 0;
    virtual void setResult(const services::SharedPtr<Result>& res) = 0;

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
        _result = services::SharedPtr<Result>(new kernel_function::Result());
        parameterBase = NULL;
        inputBase = NULL;
    }
    virtual KernelIface * cloneImpl() const DAAL_C11_OVERRIDE = 0;
    services::SharedPtr<Result> _result;
};
/** @} */
} // namespace interface1
using interface1::KernelIface;

} // namespace kernel_function
} // namespace algorithm
} // namespace daal
#endif
