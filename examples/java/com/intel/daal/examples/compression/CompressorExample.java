/* file: CompressorExample.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

/*
 //  Content:
 //     Java example of a compressor
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-COMPRESSOREXAMPLE">
 * @example CompressorExample.java
 */

package com.intel.daal.examples.compression;

import java.io.RandomAccessFile;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.zip.CRC32;
import java.util.zip.Checksum;

import com.intel.daal.data_management.compression.CompressionLevel;
import com.intel.daal.data_management.compression.zlib.ZlibCompressor;
import com.intel.daal.data_management.compression.zlib.ZlibDecompressor;
import com.intel.daal.services.DaalContext;

class CompressorExample {

    /* Queue for sending and receiving compressed data blocks */
    static Queue<byte[]> sendReceiveQueue = new LinkedList<byte[]>();

    private static final long maxDataBlockSize = 16384; /* Maximum size of a data block */

    static long compressedSize    = 0; /* Actual size of a compressed block */
    static long receivedSize      = 0; /* Actual data size in receivedDataStream */
    static long availableDataSize = 0; /* Size of the data not processed in sentDataStream */

    private static final String dataset = "../data/batch/logitboost_train.csv";

    private static byte[] sentDataStream;        /* Data stream to compress and send */
    private static byte[] uncompressedDataBlock; /* Current block of the data stream to compress */
    private static byte[] compressedDataBlock;   /* Current compressed block of data */
    private static byte[] receivedDataStream;    /* Received uncompressed data stream */

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Read data from a file and allocate memory */
        prepareMemory();

        /* Create a compressor */
        ZlibCompressor compressor = new ZlibCompressor(context);
        compressor.parameter.setCompressionLevel(CompressionLevel.DefaultLevel);
        compressor.parameter.setGzHeader(true);

        /* Receive the next data block for compression */
        while ((uncompressedDataBlock = getDataBlock()) != null) {
            /* Associate data to compress with the compressor */
            compressor.setInputDataBlock(uncompressedDataBlock);

            /* Memory for a compressed block might not be enough to compress the input block at once */
            do {
                /* Compress uncompressedDataBlock to compressedDataBlock */
                compressor.run(compressedDataBlock, maxDataBlockSize, 0);

                /* Get the actual size of a compressed block */
                compressedSize = compressor.getUsedOutputDataBlockSize();

                /* Send the current compressed block */
                sendDataBlock(compressedDataBlock, compressedSize);
            }
            /* Check if an additional data block is needed to complete compression */
            while (compressor.isOutputDataBlockFull());
        }

        /* Create a decompressor */
        ZlibDecompressor decompressor = new ZlibDecompressor(context);
        decompressor.parameter.setGzHeader(true);

        /* Receive compressed data by blocks */
        while ((compressedDataBlock = receiveDataBlock()) != null) {
            /* Associate the compressed data with the decompressor */
            decompressor.setInputDataBlock(compressedDataBlock);

            /* Decompress an incoming block to the end of receivedDataStream */
            decompressor.run(receivedDataStream, maxDataBlockSize, receivedSize);

            /* Update the size of actual data in receivedDataStream */
            receivedSize += decompressor.getUsedOutputDataBlockSize();
        }

        /* Compute and print checksums for sentDataStream and receivedDataStream */
        printCRC32(sentDataStream, receivedDataStream);

        context.dispose();
    }

    private static void prepareMemory() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read data from a file */
        sentDataStream = readData();

        compressedDataBlock = new byte[(int) maxDataBlockSize];
        receivedDataStream = new byte[sentDataStream.length];

        /* Set the size of the data not processed in sentDataStream */
        availableDataSize = sentDataStream.length;
    }

    private static byte[] readData() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read an input file */
        RandomAccessFile file = new RandomAccessFile(dataset, "r");
        int dataLength = (int) file.length();
        byte[] sentData = new byte[dataLength];
        file.read(sentData);
        file.close();

        return sentData;
    }

    private static byte[] getDataBlock() {

        long startPosition = sentDataStream.length - availableDataSize;
        byte[] currentBlock;
        /* Copy the current uncompressed block */
        if (availableDataSize >= maxDataBlockSize) {
            currentBlock = Arrays.copyOfRange(sentDataStream, (int) startPosition,
                    (int) (startPosition + maxDataBlockSize));
            availableDataSize -= maxDataBlockSize;
        } else if ((availableDataSize < maxDataBlockSize) && (availableDataSize > 0)) {
            currentBlock = Arrays.copyOfRange(sentDataStream, (int) startPosition, sentDataStream.length);
            availableDataSize = 0;
        } else {
            return null;
        }
        return currentBlock;
    }

    private static void sendDataBlock(byte[] block, long size) {

        /* Copy an incoming block to the current compressed block */
        byte[] currentBlock = Arrays.copyOf(block, (int) size);

        /* Push the current compressed block to the queue */
        sendReceiveQueue.add(currentBlock);
    }

    private static byte[] receiveDataBlock() {

        byte[] currentBlock;

        /* Receive the current compressed block from the queue or stop at the end */
        if ((currentBlock = sendReceiveQueue.poll()) == null) {
            return null;
        }
        return currentBlock;
    }

    private static void printCRC32(byte[] sentData, byte[] receivedData) {

        /* Compute checksums for full input data and full received data */
        Checksum crcSentDataStream = new CRC32();
        crcSentDataStream.update(sentData, 0, sentData.length);
        Checksum crcReceivedDataStream = new CRC32();
        crcReceivedDataStream.update(receivedData, 0, (int) receivedSize);

        System.out.println("\nCompression example program results:\n");

        System.out.println("Input data checksum:    0x" + Integer.toHexString((int) crcSentDataStream.getValue()));
        System.out.println("Received data checksum: 0x" + Integer.toHexString((int) crcReceivedDataStream.getValue()));

        if (sentData.length != receivedSize) {
            System.out.println("ERROR: Received data size (" + receivedSize + ") mismatches with the sent data size ("
                    + sentData.length + ")");
        } else if (crcSentDataStream.getValue() != crcReceivedDataStream.getValue()) {
            System.out.println("ERROR: Received data CRC mismatches with the sent data CRC");
        } else {
            System.out.println("OK: Received data CRC matches with the sent data CRC");
        }
    }
}
