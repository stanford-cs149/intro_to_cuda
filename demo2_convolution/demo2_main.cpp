#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

void my_launcher(bool useSharedMemory);
void printCudaInfo();

int main(int argc, char** argv)
{
    printCudaInfo();
    my_launcher(true);
    return 0;
}
