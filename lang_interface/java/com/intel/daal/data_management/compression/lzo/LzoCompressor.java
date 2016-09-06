/* file: LzoCompressor.java */
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

package com.intel.daal.data_management.compression.lzo;

import com.intel.daal.data_management.compression.CompressionMethod;
import com.intel.daal.data_management.compression.Compressor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__LZO__LZOCOMPRESSOR"></a>
 *
 * @brief Specialization of the Compressor class for the LZO compression method
 * \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a>
 *
 * @par References
 *      - @ref LzoCompressionParameter class
 */
public class LzoCompressor extends Compressor {
    public LzoCompressionParameter parameter; /*!< LZO compression parameters */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * LzoCompressor constructor
    */
    public LzoCompressor(DaalContext context) {
        super(context, CompressionMethod.lzo);
        parameter = new LzoCompressionParameter(context,
                cInitParameter(this.cObject, CompressionMethod.lzo.getValue()));
    }
}
