/* * * * * * * * * * * * * * * * * * * * *
 *   File:     test.cpp
 *   Author:   Li Kunyun
 *   group:    CDCS-HPC
 *   Time:     2021-07-28
 * * * * * * * * * * * * * * * * * * * * * */
#include <iostream>
using namespace std;

#include <type_traits>
#include <vector>

#include "ChipSum.hpp"

#include <cstdlib>
#include <ctime>

#include <KokkosKernels_IOUtils.hpp>



int main(int argc, char *argv[]) {
    
    ChipSum::Common::Init(argc, argv);
    {
        int M = 0;
        for (int i=0; i<60; i++){
            M += 50;
            int K = M;
            int N = M;

            double *A1 = static_cast<double *>(std::malloc(M*K * sizeof(double)));

            for(int i=0;i<M*K;++i)
            {
                A1[i] = double(rand()) / RAND_MAX;
            }
            Matrix A(M, K, A1);
            //A.Print();

            double *A2 = static_cast<double *>(std::malloc(K*N * sizeof(double)));

            for(int i=0;i<K*N;++i)
            {
                A2[i] = double(rand()) / RAND_MAX;
            }
            Matrix B(K, N, A2);
            //B.Print();

            double *A3 = static_cast<double *>(std::malloc(M*N * sizeof(double)));

            for(int i=0;i<M*N;++i)
            {
                A3[i] = double(0);
            }
            Matrix C(M, N, A3);
            //C.Print();


            //A.GEMM(B, C);
            //C.Print();

            //(A*B*3).Print();
            
            int repeat = 1000;
            /// \brief 暂时用Kokkos的Timer充数吧
            Kokkos::Timer timer;
            for(int i=0;i<repeat;++i){
                A.GEMM(B, C);
            }
            Kokkos::fence();
            double time = timer.seconds();
            

            /// \brief 带宽计算公式
            double Gbytes = repeat*1.0e-9*(2.0*M*N*K - M*K)/time;
            /* cout<<"---------------------ChipSum Perf Test"
                "---------------------"<<endl;  
            cout<<M<<endl;
            cout<<"Dense matrix GEMM performance : "<<Gbytes<<" GFlops"<<endl; */
            if(i==0){
                cout<<"---------------------ChipSum Perf Test"
                    "---------------------"<<endl
                    <<"Matrix size, Matrix size, GFlops: "<<endl;
            }
            cout<<setiosflags(ios::left)<<setw(15)<<M*K<<setw(12)<<K*N<<Gbytes<<endl;
            
            std::free(A1);
            std::free(A2);
            std::free(A3);
        }
    }
    ChipSum::Common::Finalize();
}