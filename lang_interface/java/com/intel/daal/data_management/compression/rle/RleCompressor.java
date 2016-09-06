/* file: RleCompressor.java */
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

package com.intel.daal.data_management.compression.rle;

import com.intel.daal.data_management.compression.CompressionMethod;
import com.intel.daal.data_management.compression.Compressor;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__RLE__RLECOMPRESSOR"></a>
 *
 * @brief Specialization of the Compressor class for the RLE compression method
 * \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a>
 *
 * @par References
 *      - @ref RleCompressionParameter class
 */
public class RleCompressor extends Compressor {
    public RleCompressionParameter parameter; /*!< RLE compression parameters */

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * RleCompressor constructor
    */
    public RleCompressor(DaalContext context) {
        super(context, CompressionMethod.rle);
        parameter = new RleCompressionParameter(context,
                cInitParameter(this.cObject, CompressionMethod.rle.getValue()));
    }
}
