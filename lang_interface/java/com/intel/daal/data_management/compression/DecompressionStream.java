/* file: DecompressionStream.java */
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
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__DECOMPRESSIONSTREAM"></a>
 * @brief The class that provides methods for decompressing the input compressed data arriving by the blocks.
 * \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a>
 *
 * @par References
 *      - @ref Decompressor class
 */
public class DecompressionStream extends ContextClient {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * DecompressionStream constructor
     * @param context      Context to manage decompression algorithm
     * @param decompressor Decompressor object used for the decompression
     */
    public DecompressionStream(DaalContext context, Decompressor decompressor) {
        super(context);
        this.cObject = cInit(decompressor.cObject, 64 * 1024);
    }

    /**
     * DecompressionStream constructor
     * @param context      Context to manage decompression algorithm
     * @param decompressor Decompressor object used for the decompression
     * @param minSize Minimal size of the internal data blocks
     */
    public DecompressionStream(DaalContext context, Decompressor decompressor, long minSize) {
        super(context);
        this.cObject = cInit(decompressor.cObject, minSize);
    }

    /**
     * Writes next compressed data block to the DecompressionStream and decompresses it
     * @param inBlock  Data block to be decompressed
     * @param inSize  Size of the data block to be decompressed in bytes
     */
    public void add(byte[] inBlock, long inSize) {
        cAdd(this.cObject, inBlock, inSize);
    }

    /**
     * Writes next compressed data block to the DecompressionStream and decompresses it
     * @param inBlock  Data block to be decompressed
     */
    public void add(byte[] inBlock) {
        add(inBlock, inBlock.length);
    }

    /**
     * Returns size of decompressed data stored in the DecompressionStream
     * @return Size in bytes
     */
    public long getDecompressedDataSize() {
        return cGetDecompressedDataSize(this.cObject);
    }

    /**
     * Copies decompressed data stored in the DecompressionStream to external array
     * @param outBlock Array where decompressed data is stored
     * @param outSize Number of bytes available in external memory
     * @return Size of copied data in bytes
     */
    public long copyDecompressedArray(byte[] outBlock, long outSize) {
        return cCopyDecompressedArray(this.cObject, outBlock, outSize);
    }

    /**
     * Copies decompressed data stored in the DecompressionStream to external array
     * @param outBlock Array where decompressed data is stored
     * @return Size of copied data in bytes
     */
    public long copyDecompressedArray(byte[] outBlock) {
        return copyDecompressedArray(outBlock, outBlock.length);
    }

    /**
     * Releases memory allocated for the native DecompressionStream object
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

    private native long cGetDecompressedDataSize(long strmAddr);

    private native long cCopyDecompressedArray(long strmAddr, byte[] block, long size);
}
