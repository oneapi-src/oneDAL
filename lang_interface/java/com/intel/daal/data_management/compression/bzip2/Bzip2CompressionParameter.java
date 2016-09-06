/* file: Bzip2CompressionParameter.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.data_management.compression.bzip2;

import com.intel.daal.data_management.compression.CompressionParameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__BZIP2__BZIP2COMPRESSIONPARAMETER"></a>
 *
 * @brief Parameter for the BZIP2 compression and decompression,
 *                 CompressionLevel.defaultLevel is equal to BZIP2 compression level 9
 *
 * @par Enumerations
 *      - @ref CompressionLevel - %Compression levels enumeration
 */
public class Bzip2CompressionParameter extends CompressionParameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * Bzip2CompressionParameter constructor
    */
    public Bzip2CompressionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

}
