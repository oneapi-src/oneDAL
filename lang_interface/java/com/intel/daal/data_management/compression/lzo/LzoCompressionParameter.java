/* file: LzoCompressionParameter.java */
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
package com.intel.daal.data_management.compression.lzo;

import com.intel.daal.utils.*;
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
        LibUtils.loadLibrary();
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
/** @} */
