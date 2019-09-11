/* file: service_rng_common.h */
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
//  Common RNG classes.
//--
*/

#ifndef __SERVICE_RNG_COMMON_H__
#define __SERVICE_RNG_COMMON_H__

namespace daal
{
namespace internal
{

template<CpuType cpu>
class BaseRNGIface
{
public:
    virtual int getStateSize() const = 0;
    virtual int saveState(void* dest) const = 0;
    virtual int loadState(const void* src) = 0;
    virtual int leapfrog(size_t threadNum, size_t nThreads) = 0;
    virtual int skipAhead(size_t nSkip) = 0;
    virtual void *getState() = 0;
};

} // namespace internal
} // namespace daal

#endif
