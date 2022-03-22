/* * * * * * * * * * * * * * * * * * * * *
 *   File:     lu.cpp
 *   Author:   Zhou Xingbin
 *   group:    CDCS-HPC
 *   Time:     2022-03-21
 * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
using namespace std;
#include "../ChipSum.hpp"

int main(int argc, char *argv[]) {
    
    ChipSum::Common::Init(argc, argv);
    {
        //Matrix LU
        int M = 5;
        int N = 4;
        CSFloat alpha = 1.0;
        CSFloat *a = static_cast<CSFloat *>(std::malloc(M*M * sizeof(CSFloat)));
        CSFloat *b = static_cast<CSFloat *>(std::malloc(M*N * sizeof(CSFloat)));
        for(int i=0;i<M;++i)
          for(int j=0;j<M;++j)
        {
            if(j>=i) 
              a[i*M+j] = CSFloat(rand()) / RAND_MAX;
            else
              a[i*M+j] = 0.0;
        }
        for(int i=0;i<M*N;++i)
            b[i] = CSFloat(rand()) / RAND_MAX;
        Matrix A(M, M, a);
        Matrix B(M, N, b);

        std::cout<<"origin matrix A:"<<std::endl;
        A.Print();
        std::cout<<"origin matrix B:"<<std::endl;
        B.Print();
        // B.TRMM(A,alpha,"L","U","N","N");
        B.TRMM(A,alpha,"L","U");
        std::cout<<"TRMM:X"<<std::endl;
        B.Print();   
        

        std::free(a);
        std::free(b);

    }
    ChipSum::Common::Finalize();
}