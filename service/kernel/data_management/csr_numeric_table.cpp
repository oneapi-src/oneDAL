/** file csr_numeric_table.cpp */
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

#include "csr_numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{


#define DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(T)                                                  \
template<>                                                                                          \
CSRBlockDescriptor<T>::CSRBlockDescriptor() : _rows_capacity(0), _values_capacity(0),               \
    _ncols(0), _nrows(0), _rowsOffset(0), _rwFlag(0), _rawPtr(0), _pPtr(0), _nvalues(0) {}

DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(float         )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(double        )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(int           )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(unsigned int  )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(DAAL_INT64    )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(DAAL_UINT64   )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(char          )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(unsigned char )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(short         )
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(unsigned short)
DAAL_IMPL_CSRBLOCKDESCRIPTORCONSTRUCTOR(unsigned long )

}
}
}
