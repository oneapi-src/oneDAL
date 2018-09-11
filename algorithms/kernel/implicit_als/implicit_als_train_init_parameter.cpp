/* file: implicit_als_train_init_parameter.cpp */
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
//  Implementation of auxiliary implicit als methods.
//--
*/

#include "implicit_als_train_init_parameter.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace implicit_als
{
namespace training
{
namespace init
{
namespace internal
{
SharedPtr<HomogenNumericTable<int> > getPartition(const init::DistributedParameter *parameter, services::Status &st)
{
    NumericTable *partitionTable = parameter->partition.get();
    size_t nRows = partitionTable->getNumberOfRows();
    size_t nParts = nRows - 1;
    BlockDescriptor<int> block;
    if (nRows == 1)
    {
        partitionTable->getBlockOfRows(0, nRows, readOnly, block);
        nParts = *(block.getBlockPtr());
        partitionTable->releaseBlockOfRows(block);
    }
    SharedPtr<HomogenNumericTable<int> > nt = HomogenNumericTable<int>::create(1, nParts + 1, NumericTable::doAllocate, st);
    if (!st) return nt;

    int *partition = nt->getArray();
    if (nRows == 1)
    {
        size_t nUsersInPart = parameter->fullNUsers / nParts;
        partition[0] = 0;
        for (size_t i = 1; i < nParts; i++)
        {
            partition[i] = partition[i - 1] + nUsersInPart;
        }
        partition[nParts] = parameter->fullNUsers;
    }
    else
    {
        partitionTable->getBlockOfRows(0, nRows, readOnly, block);
        int *srcPartition = block.getBlockPtr();
        for (size_t i = 0; i < nParts + 1; i++)
        {
            partition[i] = srcPartition[i];
        }
        partitionTable->releaseBlockOfRows(block);
    }
    return nt;
}

}// namespace internal
}// namespace init
}// namespace training
}// namespace implicit_als
}// namespace algorithms
}// namespace daal
