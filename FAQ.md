# Frequently Asked Questions
This document provides answers to frequently asked questions about the Intel(R) Data Analytics Acceleration Library (Intel(R) DAAL). Let us know if you have a question that is not covered here via [GitHub issues][gh-issues] or [Intel(R) DAAL Forum][daal-forum].

## Building Intel(R) DAAL from sources
> During the compilation of Java binding I got the following problem
> ```
> fatal error: jni.h: No such file or directory
>  #include <jni.h>
>                  ^
> compilation terminated.
> ```

The cause of this problem is absence of the path to [JNI][jni-wiki] headers in compiler's environment. It can be resolved by adding the required paths to the `CPATH` (on Linux\* and MacOS\*) or `INCLUDE` (on Windows\*) environment variable.
```
export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/linux:$CPATH     # for Linux*
export CPATH=$JAVA_HOME/include:$JAVA_HOME/include/darwin:$CPATH    # for MacOS*
set INCLUDE=%JAVA_HOME%/include;%JAVA_HOME%/include/win32;%INCLUDE% # for Windows*
```

### Windows\* specific questions
> During the build process I can see errors reported by the `find` util. Any ideas on what can cause them?

It is likely caused by execution of the MSYS2* `find` command on FAT32 file system. Building Intel DAAL on NTFS should not cause these issues.

> What can cause Makefile fail with "fork: Resource temporarily unavailable" error?

This error can be caused by execution of the MSYS2* `make` util with a large number of threads. You can reduce the number of threads to solve this issue. Try using make with a smaller number of threads, e.g. `make daal -j 4 PLAT=win32e`.

<!-- Links -->
[daal-forum]: https://software.intel.com/en-us/forums/intel-data-analytics-acceleration-library
[gh-issues]:  https://github.com/01org/daal/issues
[jni-wiki]:   https://en.wikipedia.org/wiki/Java_Native_Interface
