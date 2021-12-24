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
        for (int i=0; i<150; i++){
            M += 50;
            int N = M;

            double *A1 = static_cast<double *>(std::malloc(M*N * sizeof(double)));

            for(int i=0;i<M*N;++i)
            {
                A1[i] = double(rand()) / RAND_MAX;
            }
            Matrix A(M, N, A1);

            double *A2 = static_cast<double *>(std::malloc(N * sizeof(double)));

            for(int i=0;i<N;++i)
            {
                A2[i] = double(rand()) / RAND_MAX;
            }
            Vector x(N, A2);

            double *A3 = static_cast<double *>(std::malloc(M * sizeof(double)));

            for(int i=0;i<M;++i)
            {
                A3[i] = double(0);
            }
            Vector r(M, A3);

            
            int repeat = 1000;
            /// \brief 暂时用Kokkos的Timer充数吧
            Kokkos::Timer timer;
            for(int i=0;i<repeat;++i){
                A.GEMV(x, r);
            }
            Kokkos::fence();
            double time = timer.seconds();
            

            /// \brief 带宽计算公式
            double Gbytes = repeat*1.0e-9*(2.0*M*N - M)/time;
            /* cout<<"---------------------ChipSum Perf Test"
                "---------------------"<<endl;  
            cout<<M<<endl;
            cout<<"Dense matrix GEMV performance : "<<Gbytes<<" GFlops"<<endl; */
            if(i==0){
                cout<<"---------------------ChipSum Perf Test"
                    "---------------------"<<endl
                    <<"Matrix size, Vector size, GFlops: "<<endl;
            }
            cout<<setiosflags(ios::left)<<setw(15)<<M*M<<setw(12)<<N<<Gbytes<<endl;
            std::free(A1);
            std::free(A2);
            std::free(A3);
        }
    }
    ChipSum::Common::Finalize();
}