/* file: kernel_function_rbf.cpp */
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
//  Implementation of kernel function algorithm and types methods.
//--
*/

#include "kernel_function_types_rbf.h"
#include "service_defines.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kernel_function
{
namespace rbf
{
namespace interface1
{
Parameter::Parameter(double sigma) : ParameterBase(), sigma(sigma) {}

Input::Input() : kernel_function::Input() {}
Input::Input(const Input& other) : kernel_function::Input(other){}

/**
 * Checks input objects of the RBF kernel algorithm
 * \param[in] par     %Input objects of the algorithm
 * \param[in] method   Computation method of the algorithm
 */
Status Input::check(const daal::algorithms::Parameter *par, int method) const
{
    switch(method)
    {
    case fastCSR:
        return checkCSR();
    case defaultDense:
        return checkDense();
    default:
        DAAL_ASSERT(false);
        break;
    }

    return services::Status();
}

}// namespace interface1
}// namespace rbf
}// namespace kernel_function
}// namespace algorithms
}// namespace daal
