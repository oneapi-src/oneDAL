wget https://sourceware.org/pub/valgrind/valgrind-3.22.0.tar.bz2
tar -xjf valgrind-3.22.0.tar.bz2
cd valgrind-3.22.0
./configure --enable-only64bit
make
make install
valgrind ls -l