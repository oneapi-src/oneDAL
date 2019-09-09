/* file: CompressionOnline.java */
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
 //     Java example of compression in the online processing mode
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-COMPRESSIONONLINE">
 * @example CompressionOnline.java
 */

package com.intel.daal.examples.compression;

import java.io.RandomAccessFile;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.zip.CRC32;
import java.util.zip.Checksum;

import com.intel.daal.data_management.compression.*;
import com.intel.daal.data_management.compression.zlib.*;
import com.intel.daal.services.DaalContext;

class CompressionOnline {

    static Queue<byte[]> sendReceiveQueue = new LinkedList<byte[]>(); /* Queue for sending and receiving compressed data blocks */

    private static final long maxDataBlockSize     = 16384; /* Maximum size of a data block */
    private static final long userDefinedBlockSize = 7000;  /* Size for read data from a decompression stream */

    static long receivedSize      = 0; /* Actual data size in receivedDataStream */
    static long availableDataSize = 0; /* Data size not processed in sentDataStream */

    private static final String dataset = "../data/online/logitboost_train.csv";

    private static byte[] sentDataStream;        /* Data stream to compress and send */
    private static byte[] uncompressedDataBlock; /* Current block of the data stream to compress */
    private static byte[] compressedDataBlock;   /* Current compressed block of data */
    private static byte[] readDataBlock;         /* Current data block to store decompressed data */
    private static byte[] receivedDataStream;    /* Received uncompressed data stream */

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Read data from a file and allocate memory */
        prepareMemory();

        /* Create a compressor */
        ZlibCompressor compressor = new ZlibCompressor(context);
        compressor.parameter.setCompressionLevel(CompressionLevel.DefaultLevel);
        compressor.parameter.setGzHeader(true);

        /* Create a stream for compression */
        CompressionStream compressionStream = new CompressionStream(context, compressor);

        /* Receive the next data block for compression */
        while ((uncompressedDataBlock = getDataBlock()) != null) {
            /* Put a data block to compressionStream and compress if needed */
            compressionStream.add(uncompressedDataBlock);

            /* Get the size of the compressed data */
            compressedDataBlock = new byte[(int) compressionStream.getCompressedDataSize()];

            /* Store the compressed data in compressedDataBlock*/
            compressionStream.copyCompressedArray(compressedDataBlock);

            /* Send the current compressed block */
            sendDataBlock(compressedDataBlock, compressedDataBlock.length);
        }

        /* Create a decompressor */
        ZlibDecompressor decompressor = new ZlibDecompressor(context);
        decompressor.parameter.setGzHeader(true);

        /* Create a stream for decompression */
        DecompressionStream decompressionStream = new DecompressionStream(context, decompressor);

        /* Actual size of the decompressed data currently read from decompressionStream */
        long readSize = 0;

        /* Receive compressed data by blocks */
        while ((compressedDataBlock = receiveDataBlock()) != null) {
            /* Write the received block to decompressionStream */
            decompressionStream.add(compressedDataBlock);

            /* Asynchronous read from decompressionStream */
            do {
                /* Read userDefinedBlockSize bytes from decompressionStream to readDataBlock */
                readSize = decompressionStream.copyDecompressedArray(readDataBlock);

                /* Update the actual data size in receivedDataStream */
                System.arraycopy(readDataBlock, 0, receivedDataStream, (int) receivedSize, (int) readSize);

                /* Update the actual data size in receivedDataStream */
                receivedSize += readSize;
            } while (readSize != 0);
        }

        /* Compute and print checksums for sentDataStream and receivedDataStream */
        printCRC32(sentDataStream, receivedDataStream);

        context.dispose();
    }

    private static void prepareMemory() throws java.io.FileNotFoundException, java.io.IOException {

        /* Read data from a file */
        sentDataStream = readData();

        receivedDataStream = new byte[sentDataStream.length];

        /* Set the size of the data not processed in sentDataStream */
        availableDataSize = sentDataStream.length;

        readDataBlock = new byte[(int) userDefinedBlockSize];
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
