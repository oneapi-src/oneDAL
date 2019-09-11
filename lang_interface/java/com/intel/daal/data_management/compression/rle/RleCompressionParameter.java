/* file: RleCompressionParameter.java */
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

/**
 * @ingroup data_compression
 * @{
 */
package com.intel.daal.data_management.compression.rle;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
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
/** @} */
