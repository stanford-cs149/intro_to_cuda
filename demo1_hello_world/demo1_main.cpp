#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

void my_launcher();
void my_launcher_2D();
void printCudaInfo();

int main(int argc, char** argv)
{
    printCudaInfo();
    
    // my_launcher();
    my_launcher_2D();


    return 0;
}
