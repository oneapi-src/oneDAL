/* file: LzoCompressionParameter.java */
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

import com.intel.daal.data_management.compression.CompressionParameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__LZO__LZOCOMPRESSIONPARAMETER"></a>
 *
 * @brief Parameter for the LZO compression and decompression
 * LZO compressed block header consists of four sections: 1) optional, 2) uncompressed data size(4 bytes),
 * 3) compressed data size(4 bytes), 4) optional.
 *
 * @par Enumerations
 *      - @ref CompressionLevel - %Compression levels enumeration
 */
public class LzoCompressionParameter extends CompressionParameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public LzoCompressionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Sets size of section 1) of LZO compressed block header in bytes
     *  @param preHeadBytes Size of section 1) of LZO compressed block header in bytes
     */
    public void setPreHeadBytes(long preHeadBytes) {
        cSetPreHeadBytes(this.cObject, preHeadBytes);
    }

    /**
     *  Returns size of section 1) of LZO compressed block header in bytes
     *  @return Size of section 1) of LZO compressed block header in bytes
     */
    public long getPreHeadBytes() {
        return cGetPreHeadBytes(this.cObject);
    }

    /**
     *  Sets size of section 4) of LZO compressed block header in bytes
     *  @param postHeadBytes Size of section 4) of LZO compressed block header in bytes
     */
    public void setPostHeadBytes(long postHeadBytes) {
        cSetPostHeadBytes(this.cObject, postHeadBytes);
    }

    /**
     *  Returns size of section 4) of LZO compressed block header in bytes
     *  @return Size of section 4) of LZO compressed block header in bytes
     */
    public long getPostHeadBytes() {
        return cGetPostHeadBytes(this.cObject);
    }

    private native void cSetPreHeadBytes(long parAddr, long preHeadBytes);

    private native long cGetPreHeadBytes(long parAddr);

    private native void cSetPostHeadBytes(long parAddr, long postHeadBytes);

    private native long cGetPostHeadBytes(long parAddr);
}
