#ifndef __CHIPSUM_DENSEMAT_SERIAL_IMPL_HPP__
#define __CHIPSUM_DENSEMAT_SERIAL_IMPL_HPP__


#include <vector>
#include <cassert>
#include <fstream>

#include "../numeric_traits.hpp"
#include "../../chipsum_macro.h"



namespace ChipSum {

namespace  Numeric {


template<typename ScalarType,typename SizeType,typename ...Props>
struct DenseMatrix_Traits<ScalarType,SizeType,ChipSum::Backend::BuiltinSerial,Props...>
        : public Operator_Traits<ScalarType,SizeType,ChipSum::Backend::BuiltinSerial,Props...>{



    using matrix_type = std::vector<ScalarType>;

    using size_type = std::size_t;


};


namespace  Impl {
namespace  DenseMat
{



template <typename ScalarType,typename SizeType,typename ...Props>
CHIPSUM_FUNCTION_INLINE void Create(const std::size_t M,
                                    const std::size_t N,
                                    std::vector<ScalarType>& mat)
{



    mat = std::vector<ScalarType>(M*N);

}



template <typename ScalarType,typename SizeType,typename ...Props>
CHIPSUM_FUNCTION_INLINE void Fill(const std::size_t M,
                                  const std::size_t N,
                                  ScalarType* src,
                                  std::vector<ScalarType>& dst)
{
    dst = std::vector<ScalarType>(src,src+M*N);
}

template <typename ScalarType,typename SizeType,typename ...Props>
CHIPSUM_FUNCTION_INLINE void Mult(const std::size_t M,
                                  const std::size_t N,
                                  const std::vector<ScalarType>& A,
                                  const std::vector<ScalarType>& x,
                                  std::vector<ScalarType>& b)
{
    assert(A.size()==M*N && x.size()==N);
    if(b.size()!=M)b.resize(M);

    for(std::size_t i=0;i<M;++i)b[i] = 0;

    for(std::size_t i=0;i<M;++i){
        for(std::size_t j=0;j<N;++j){
            b[i] += A[i*N+j] * x[j];
        }
    }
}


template <typename ScalarType,typename SizeType,typename ...Props>
CHIPSUM_FUNCTION_INLINE void Mult(const std::size_t M,
                                  const std::size_t N,
                                  const std::size_t K,
                                  const std::vector<ScalarType>& A,
                                  const std::vector<ScalarType>& B,
                                  std::vector<ScalarType>& C)
{
    assert(A.size()==M*K && B.size()==N*K);
    if(C.size()!=M*N)C.resize(M*N);

    for(std::size_t i=0;i<C.size();++i)C[i] = 0;

    for(std::size_t i=0;i<M;++i){
        for(std::size_t j=0;j<N;++j){
            C[i] += A[i*N+j] * B[j];
        }
    }
}



template <typename ScalarType,typename SizeType,typename ...Props>
CHIPSUM_FUNCTION_INLINE void Scal(ScalarType alpha,
                                  const std::size_t M,
                                  const std::size_t N,
                                  std::vector<ScalarType>& mat)
{
    assert(mat.size() == M*N);
    for(std::size_t i=0;i<mat.size();++i)
    {
        mat[i] *= alpha;
    }
}


template <typename ScalarType,typename SizeType,typename ...Props>
CHIPSUM_FUNCTION_INLINE ScalarType& GetItem(const std::size_t i,
                                            const std::size_t j,
                                            const std::size_t M,
                                            const std::size_t N,
                                            std::vector<ScalarType>& mat
                                            )
{
    assert(i<M && j<N);
    return mat[i*N+j];
}


template <typename ScalarType,typename SizeType,typename ...Props>
CHIPSUM_FUNCTION_INLINE void Print(const std::size_t M,
                                   const std::size_t N,
                                   std::vector<ScalarType>& mat,
                                   std::ostream &out)
{

    for(std::size_t i=0;i<M;++i)
    {
        out<<" "<<"[";
        for(std::size_t j=0;j<M-1;++j)
        {
            out<<mat[i*N+j]<<", ";
        }
        out<<mat[i*N+M-1]<<"]"<<endl;

    }
    out<<endl;

}



} // namespace DenseMat
} // namespace Impl
} // namespace Numeric
} // namespace ChipSum



#endif // __CHIPSUM_DENSEMAT_BLAS_IMPL_HPP__
