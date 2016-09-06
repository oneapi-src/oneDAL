/* file: ZlibCompressionParameter.java */
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

package com.intel.daal.data_management.compression.zlib;

import com.intel.daal.data_management.compression.CompressionParameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__ZLIB__ZLIBCOMPRESSIONPARAMETER"></a>
 *
 * @brief Parameter for ZLIB compression and decompression
 *
 * @par Enumerations
 *      - @ref CompressionLevel - %Compression levels enumeration
 */
public class ZlibCompressionParameter extends CompressionParameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * ZlibCompressionParameter constructor
    */
    public ZlibCompressionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Sets optional GZIP header flag
     *  @param gzHeader Optional GZIP header flag. True if simple GZIP header is included, false otherwise
     */
    public void setGzHeader(boolean gzHeader) {
        cSetGzHeader(this.cObject, gzHeader);
    }

    /**
     *  Returns optional GZIP header flag
     *  @return Optional GZIP header flag. True if simple GZIP header is included, false otherwise
     */
    public boolean getGzHeader() {
        return cGetGzHeader(this.cObject);
    }

    private native void cSetGzHeader(long parAddr, boolean gzHeader);

    private native boolean cGetGzHeader(long parAddr);
}
