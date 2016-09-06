/* file: RleCompressionParameter.java */
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

import com.intel.daal.data_management.compression.CompressionParameter;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-SERVICE__COMPRESSION__RLE__RLECOMPRESSIONPARAMETER"></a>
 *
 * @brief Parameter for the RLE encoding and decoding.
 * RLE encoded block may contain header that consists of two sections: decoded data size(4 bytes), and encoded data size(4 bytes)
 *
 */
public class RleCompressionParameter extends CompressionParameter {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
    * RleCompressionParameter constructor
    */
    public RleCompressionParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Sets flag which indicates whether there is an RLE block header
     *  @param isBlockHeader Flag which indicates whether there is a RLE block header. True if the RLE block header is present, false otherwise
     */
    public void setBlockHeader(boolean isBlockHeader) {
        cSetBlockHeader(this.cObject, isBlockHeader);
    }

    /**
     *  Returns RLE block header presence flag
     *  @return RLE block header presence flag. True if the RLE block header is present, false otherwise
     */
    public boolean getBlockHeader() {
        return cGetBlockHeader(this.cObject);
    }

    private native void cSetBlockHeader(long parAddr, boolean gzHeader);

    private native boolean cGetBlockHeader(long parAddr);
}
