#include <iostream>
int empty() {
    volatile int three = 1 + 2;
    return three;
}