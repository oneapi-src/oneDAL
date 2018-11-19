/* file: kernel_function_types_rbf.h */
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

#ifndef __KERNEL_FUNCTION_TYPES_RBF_H__
#define __KERNEL_FUNCTION_TYPES_RBF_H__

#include "algorithms/kernel_function/kernel_function_types.h"

namespace daal
{
namespace algorithms
{
/**
 * @defgroup kernel_function_rbf Radial Basis Function Kernel
 * \copydoc daal::algorithms::kernel_function::rbf
 * @ingroup kernel_function
 * @{
 */
/**
 * \brief Contains classes for computing kernel functions
 */
namespace kernel_function
{
/**
 * \brief Contains classes for computing the radial basis function (RBF) kernel
 */
namespace rbf
{
/**
 * <a name="DAAL-ENUM-ALGORITHMS__KERNEL_FUNCTION__RBF__METHOD"></a>
 * Method for computing  kernel functions
 */
enum Method
{
    defaultDense = 0,    /*!< Default method for computing the RBF kernel */
    fastCSR = 1          /*!< Fast: performance-oriented method. Works with Compressed Sparse Rows (CSR) numeric tables */
};

/**
 * \brief Contains version 1.0 of the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL) interface.
 */
namespace interface1
{
/**
 * <a name="DAAL-STRUCT-ALGORITHMS__KERNEL_FUNCTION__RBF__PARAMETER"></a>
 * \brief Parameters for the radial basis function (RBF) kernel
 *
 * \snippet kernel_function/kernel_function_types_rbf.h RBF input object source code
 */
/* [RBF input object source code] */
struct DAAL_EXPORT Parameter : public ParameterBase
{
    Parameter(double sigma = 1.0);
    double sigma;   /*!< RBF kernel coefficient */
};
/* [RBF input object source code] */

/**
 * <a name="DAAL-CLASS-ALGORITHMS__KERNEL_FUNCTION__RBF__INPUT"></a>
 * \brief %Input objects for the RBF kernel algorithm
 */
class DAAL_EXPORT Input : public kernel_function::Input
{
public:
    Input();
    Input(const Input& other);

    virtual ~Input() {}

    /**
    * Checks input objects of the RBF kernel algorithm
    * \param[in] par     %Input objects of the algorithm
    * \param[in] method   Computation method of the algorithm
    */
    services::Status check(const daal::algorithms::Parameter *par, int method) const DAAL_C11_OVERRIDE;
};
/** @} */
} // namespace interface1
using interface1::Input;
using interface1::Parameter;

} // rbf
} // namespace kernel_function
} // namespace algorithms
} // namespace daal
#endif
