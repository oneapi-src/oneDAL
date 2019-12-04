/* file: compression_types.i */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include <jni.h>

#include "daal.h"

#include "com_intel_daal_data_management_compression_CompressionLevel.h"
#define DefaultLevel com_intel_daal_data_management_compression_CompressionLevel_DefaultLevelValue
#define Level0       com_intel_daal_data_management_compression_CompressionLevel_Level0Value
#define Level1       com_intel_daal_data_management_compression_CompressionLevel_Level1Value
#define Level2       com_intel_daal_data_management_compression_CompressionLevel_Level2Value
#define Level3       com_intel_daal_data_management_compression_CompressionLevel_Level3Value
#define Level4       com_intel_daal_data_management_compression_CompressionLevel_Level4Value
#define Level5       com_intel_daal_data_management_compression_CompressionLevel_Level5Value
#define Level6       com_intel_daal_data_management_compression_CompressionLevel_Level6Value
#define Level7       com_intel_daal_data_management_compression_CompressionLevel_Level7Value
#define Level8       com_intel_daal_data_management_compression_CompressionLevel_Level8Value
#define Level9       com_intel_daal_data_management_compression_CompressionLevel_Level9Value

#include "com_intel_daal_data_management_compression_CompressionMethod.h"
#define Zlib  com_intel_daal_data_management_compression_CompressionMethod_Zlib
#define Lzo   com_intel_daal_data_management_compression_CompressionMethod_Lzo
#define Rle   com_intel_daal_data_management_compression_CompressionMethod_Rle
#define Bzip2 com_intel_daal_data_management_compression_CompressionMethod_Bzip2
