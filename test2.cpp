///
/// \file     test2.cpp
/// \author   Riiiichman-Li
/// \group    CDCS-HPC
/// \date     2021-12-21
/// \brief    %stuff%
///
#include "ChipSum.hpp"
#include "chipsum/chipsum_macro.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Vector.hpp>
#include <Kokkos_Crs.hpp>


typedef Kokkos::vector<double> vector_type;


struct print_functor{
    vector_type _a;
    print_functor(vector_type a)
        :_a(a)
    {

    }

    KOKKOS_INLINE_FUNCTION void operator()(const int i)const{
        printf("%f\n",_a.d_view(i));
    }
};

int main(int argc,char* argv[])
{
    ChipSum::Common::Init(argc, argv);
    {
        vector_type v(5);
        v.push_back(3);
        v.push_back(3);
        v.push_back(3);





        v.host_to_device();

        Kokkos::parallel_for(v.size(),print_functor(v));


    }
    ChipSum::Common::Finalize();
}
