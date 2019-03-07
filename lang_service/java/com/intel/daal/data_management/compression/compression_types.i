/* file: compression_types.i */
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

#include <jni.h>

#include "JCompressionLevel.h"
#include "JCompressionMethod.h"
#include "daal.h"

#define DefaultLevel com_intel_daal_data_management_compression_CompressionLevel_DefaultLevelValue
#define Level0 com_intel_daal_data_management_compression_CompressionLevel_Level0Value
#define Level1 com_intel_daal_data_management_compression_CompressionLevel_Level1Value
#define Level2 com_intel_daal_data_management_compression_CompressionLevel_Level2Value
#define Level3 com_intel_daal_data_management_compression_CompressionLevel_Level3Value
#define Level4 com_intel_daal_data_management_compression_CompressionLevel_Level4Value
#define Level5 com_intel_daal_data_management_compression_CompressionLevel_Level5Value
#define Level6 com_intel_daal_data_management_compression_CompressionLevel_Level6Value
#define Level7 com_intel_daal_data_management_compression_CompressionLevel_Level7Value
#define Level8 com_intel_daal_data_management_compression_CompressionLevel_Level8Value
#define Level9 com_intel_daal_data_management_compression_CompressionLevel_Level9Value

#define Zlib com_intel_daal_data_management_compression_CompressionMethod_Zlib
#define Lzo com_intel_daal_data_management_compression_CompressionMethod_Lzo
#define Rle com_intel_daal_data_management_compression_CompressionMethod_Rle
#define Bzip2 com_intel_daal_data_management_compression_CompressionMethod_Bzip2
