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


#include <KokkosKernels_IOUtils.hpp>



int main(int argc, char *argv[]) {
    
    ChipSum::Common::Init(argc, argv);
    {   int N=100;
        for(int j=0; j<200; ++j){
            float *v1 = static_cast<float *>(std::malloc(N * sizeof(float)));
            float *v2 = static_cast<float *>(std::malloc(N * sizeof(float)));
            
            for (int i = 0; i < N; ++i) {
                v1[i] = float(i);
                v2[i] = float(i);
            }
            
            Vector a(N,v1); // a = {0,1,2,3,4...}
            Vector b(N,v2); // b = {0,1,2,3,4...}

            // Scalar r ;
            int repeat = 100;
            /// \brief 暂时用Kokkos的Timer
            Kokkos::Timer timer;
            for(int i=0;i<repeat;++i){
            a.Dot(b);
            }
            Kokkos::fence();
            double time = timer.seconds();
            

            /// \brief 带宽计算公式
            double Gbytes = repeat*1.0e-9*(2.0*N-1)/time;
            if(j==0){
                Kokkos::DefaultExecutionSpace::print_configuration(cout,true);
                cout<<"---------------------ChipSum dot Perf Test "
                      "---------------------"<<endl
                    <<"Vector size,GFlops :"<<endl;
            }
            cout<<setiosflags(ios::left)<<setw(12)<<N<<Gbytes<<endl;
            
            N*=1.1;          
            std::free(v1);
            std::free(v2);
        }
    }
    ChipSum::Common::Finalize();
}