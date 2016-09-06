/* file: CompressionStream.java */
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

package com.intel.daal.data_management.compression;

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__COMPRESSIONSTREAM"></a>
 * @brief The class that provides methods for compressing input raw data by the blocks. *
 * \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a>
 * @par References
 *      - @ref Compressor class
 */
public class CompressionStream extends ContextClient {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
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
