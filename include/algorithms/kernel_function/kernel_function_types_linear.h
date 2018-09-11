/* file: kernel_function_types_linear.h */
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
//  Kernel function parameter structure
//--
*/

#ifndef __KERNEL_FUNCTION_TYPES_LINEAR_H__
#define __KERNEL_FUNCTION_TYPES_LINEAR_H__

#include "algorithms/kernel_function/kernel_function_types.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup kernel_function_linear Linear Kernel
 * \copydoc daal::algorithms::kernel_function::linear
 * @ingroup kernel_function
 * @{
 */
/**
 * \brief Contains classes for computing kernel functions
 */
namespace kernel_function
{
/**
 * \brief Contains classes for computing linear kernel functions
 */
namespace linear
{

/**
 * <a name="DAAL-ENUM-ALGORITHMS__KERNEL_FUNCTION__LINEAR__METHOD"></a>
 * Method of the kernel function
 */
enum Method
{
    defaultDense = 0,    /*!< Default method for computing linear kernel functions */
    fastCSR = 1          /*!< Fast: performance-oriented method. Works with Compressed Sparse Rows (CSR) numeric tables */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__KERNEL_FUNCTION__LINEAR__PARAMETER"></a>
 * \brief Parameters for the linear kernel function k(X,Y) + b
 *
 * \snippet kernel_function/kernel_function_types_linear.h Linear input object source code
 */
/* [Linear input object source code] */
struct DAAL_EXPORT Parameter: public ParameterBase
{
    Parameter(double k = 1.0, double b = 0.0);
    double k;       /*!< Linear kernel coefficient k in the k(X,Y) + b model */
    double b;       /*!< Linear kernel coefficient b in the k(X,Y) + b model */
};
/* [Linear input object source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__LINEAR__INPUT"></a>
 * \brief %Input objects for the kernel function linear algorithm
 */
class DAAL_EXPORT Input : public kernel_function::Input
{
public:
    Input();
    Input(const Input& other);

    virtual ~Input() {}

    /**
    * Checks input objects of the kernel function linear algorithm
    * \param[in] par     %Input objects of the algorithm
    * \param[in] method  Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};
/** @} */
} // namespace interface1
using interface1::Input;
using interface1::Parameter;

} // linear
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
