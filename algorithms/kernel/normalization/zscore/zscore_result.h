/* file: zscore_result.h */
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
//  Implementation of zscore algorithm and types methods.
//--
*/

#ifndef __ZSCORE_RESULT_H__
#define __ZSCORE_RESULT_H__

#include "zscore_types.h"
#include "inner/zscore_result_v1.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace normalization
{
namespace zscore
{

namespace interface2
{

class ResultImpl : public interface1::ResultImpl
{
public:
    DAAL_CAST_OPERATOR(ResultImpl);

    ResultImpl(const size_t n) : interface1::ResultImpl(n) {}
    ResultImpl(const ResultImpl& o) : interface1::ResultImpl(o){}

    using DataCollection::operator[];
    virtual ~ResultImpl() {};

    /**
    * Allocates memory to store final results of the z-score normalization algorithms
    * \param[in] input     Input objects for the z-score normalization algorithm
    * \param[in] parameter Pointer to algorithm parameter
    *
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter);

    /**
    * Checks the correctness of the Result object
    * \param[in] in     Pointer to the input object
    * \param[in] par    Pointer to algorithm parameter
    *
    * \return Status of computations
    */
    services::Status check(const daal::algorithms::Input *in, const daal::algorithms::Parameter *par) const;
};

}// namespace interface2

}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal

#endif
