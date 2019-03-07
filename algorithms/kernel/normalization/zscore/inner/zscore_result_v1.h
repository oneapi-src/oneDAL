/* file: zscore_result_v1.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
#ifndef __ZSCORE_RESULT_V1_H__
#define __ZSCORE_RESULT_V1_H__

#include "zscore_types_v1.h"

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

namespace interface1
{

class ResultImpl : public DataCollection
{
public:
    DAAL_CAST_OPERATOR(ResultImpl);

    ResultImpl(const size_t n) : DataCollection(n) {}
    ResultImpl(const ResultImpl& o) : DataCollection(o){}
    virtual ~ResultImpl() {};

    /**
    * Allocates memory to store final results of the z-score normalization algorithms
    * \param[in] input     Input objects for the z-score normalization algorithm
    *
    * \return Status of computations
    */
    template <typename algorithmFPType>
    services::Status allocate(const daal::algorithms::Input *input);

    /**
    * Checks the correctness of the Result object
    * \param[in] in     Pointer to the input object
    *
    * \return Status of computations
    */
    services::Status check(const daal::algorithms::Input *in) const;
};

}//namespace  interface 1

}// namespace zscore
}// namespace normalization
}// namespace algorithms
}// namespace daal

#endif // __ZSCORE_RESULT_V1_H__
