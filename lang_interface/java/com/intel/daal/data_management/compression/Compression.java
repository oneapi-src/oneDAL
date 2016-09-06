/* file: Compression.java */
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

/**
 * @brief Contains classes for data compression and decompression
 */
package com.intel.daal.data_management.compression;

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__COMPRESSION__COMPRESSION"></a>
 * @brief The base class that provides methods for the compression and decompression operation
 * \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a>
 *
 * @par References
 *      - CompressionParameter class
 */
public class Compression extends ContextClient {
    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    /**
     * @brief Compression constructor
     */
    public Compression(DaalContext context) {
        super(context);
    }

    /**
     * @brief Pointer to C++ implementation of the Compression
     */
    public long cObject;

    /**
     * Associates input data block with the Compressor(or Decompressor)
     *
     * @param inBlock Data block to compress(or decompress). Must be at least srcSize+offset bytes
     * @param srcSize  The number of bytes to compress(or decompress) in the inBlock
     * @param offset   Offset in bytes, the starting position for the compression(or decompression) in inBlock
     */
    public void setInputDataBlock(byte[] inBlock, long srcSize, long offset) {
        cSetInputDataBlock(this.cObject, inBlock, srcSize, offset);
    }

    /**
     * Associates input data block with the Compressor(or Decompressor)
     *
     * @param inBlock Data block to compress(or decompress)
     * @param offset   Offset in bytes, the starting position for the compression(or decompression) in inBlock
     */
    public void setInputDataBlock(byte[] inBlock, long offset) {
        setInputDataBlock(inBlock, inBlock.length - offset, offset);
    }

    /**
     * Associates input data block with the compressor(or decompressor)
     *
     * @param inBlock Data block to compress(or decompress)
     */
    public void setInputDataBlock(byte[] inBlock) {
        setInputDataBlock(inBlock, inBlock.length, 0);
    }

    /**
     * Reports whether output data block is full after run() method was called
     *
     * @return True if output data block is full, false otherwise
     */
    public boolean isOutputDataBlockFull() {
        return cIsOutputDataBlockFull(this.cObject);
    }

    /**
     * Returns the number of bytes used after run() method was called
     *
     * @return Number of used bytes
     */
    public long getUsedOutputDataBlockSize() {
        return cGetUsedOutputDataBlockSize(this.cObject);
    }

    /**
     * Performs the compression(or decompression) of the data block
     *
     * @param outBlock Data block where compression(or decompression) results to be stored. Must be at least size+offset bytes
     * @param size       Number of bytes available in outBlock
     * @param offset     Offset in bytes, the starting position for the compression(or decompression) in outBlock
     */
    public void run(byte[] outBlock, long size, long offset) {
        cRun(this.cObject, outBlock, size, offset);
    }

    /**
     * Checks of the input stream parameters
     * @param inBlock Input data block
     * @param size     Size in bytes of the input data block
     */
    public void checkInputParams(byte[] inBlock, long size) {
        cCheckInputParams(this.cObject, inBlock, size);
    }

    /**
     * Checks of the input stream parameters
     * @param inBlock Input data block
     */
    public void checkInputParams(byte[] inBlock) {
        checkInputParams(inBlock, inBlock.length);
    }

    /**
     * Checks of the output stream parameters
     * @param outBlock Output data block
     * @param size      Size in bytes of the output data block
     */
    public void checkOutputParams(byte[] outBlock, long size) {
        cCheckOutputParams(this.cObject, outBlock, size);
    }

    /**
     * Checks of the output stream parameters
     * @param outBlock Output data block
     */
    public void checkOutputParams(byte[] outBlock) {
        checkOutputParams(outBlock, outBlock.length);
    }

    /**
     * Releases memory allocated for the native Compression object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    protected native boolean cIsOutputDataBlockFull(long compressionAddress);

    protected native long cGetUsedOutputDataBlockSize(long compressionAddress);

    protected native void cDispose(long compressionAddress);

    protected native void cSetInputDataBlock(long compressionAddress, byte[] inBlock, long size, long offset);

    protected native void cRun(long compressionAddress, byte[] outBlock, long size, long offset);

    protected native void cCheckInputParams(long compressionAddress, byte[] inBlock, long size);

    protected native void cCheckOutputParams(long compressionAddress, byte[] outBlock, long size);
}
