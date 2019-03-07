/* file: CompressionStream.java */
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
package com.intel.daal.data_management.compression;

import com.intel.daal.utils.*;
import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__COMPRESSIONSTREAM"></a>
 * @brief The class that provides methods for compressing input raw data by the blocks. *
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 * @par References
 *      - @ref Compressor class
 */
public class CompressionStream extends ContextClient {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * CompressionStream constructor
     * @param context   Context to manage created compression algorithm
     * @param compressor Compressor object used for the compression
     */
    public CompressionStream(DaalContext context, Compressor compressor) {
        super(context);
        this.cObject = cInit(compressor.cObject, 64 * 1024);
    }

    /**
     * CompressionStream constructor
     * @param context   Context to manage created compression algorithm
     * @param compressor Compressor object used for the compression
     * @param minSize Minimal size of the internal data blocks
     */
    public CompressionStream(DaalContext context, Compressor compressor, long minSize) {
        super(context);
        this.cObject = cInit(compressor.cObject, minSize);
    }

    /**
     * Writes next data block to the CompressionStream and compresses it
     * @param inBlock  Data block to be compressed
     * @param inSize   Size in bytes of the data block to be compressed
     */
    public void add(byte[] inBlock, long inSize) {
        cAdd(this.cObject, inBlock, inSize);
    }

    /**
     * Writes next data block to the CompressionStream and compresses it
     * @param inBlock  Data block to be compressed
     */
    public void add(byte[] inBlock) {
        add(inBlock, inBlock.length);
    }

    /**
     * Returns size of the compressed data stored in the CompressionStream
     * @return Size in bytes
     */
    public long getCompressedDataSize() {
        return cGetCompressedDataSize(this.cObject);
    }

    /**
     * Copies compressed data stored in the CompressionStream to external array
     * @param outBlock Array where compressed data is stored
     * @param outSize Number of bytes available in the array
     * @return Size of copied data in bytes
     */
    public long copyCompressedArray(byte[] outBlock, long outSize) {
        return cCopyCompressedArray(this.cObject, outBlock, outSize);
    }

    /**
     * Copies compressed data stored in the CompressionStream to external array
     * @param outBlock Array where compressed data is stored
     * @return Size of copied data in bytes
     */
    public long copyCompressedArray(byte[] outBlock) {
        return copyCompressedArray(outBlock, outBlock.length);
    }

    /**
     * Releases memory allocated for the native CompressionStream object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    protected native void cDispose(long strAddr);

    public long cObject;

    private native long cInit(long decomprAddr, long minSize);

    private native void cAdd(long strmAddr, byte[] block, long size);

    private native long cGetCompressedDataSize(long strmAddr);

    private native long cCopyCompressedArray(long strmAddr, byte[] block, long size);
}
/** @} */
