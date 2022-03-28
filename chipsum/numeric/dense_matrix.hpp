///
/// \file     dense_matrix.hpp
/// \author   Riiiichman-Li
/// \group    CDCS-HPC
/// \date     2021-11-01
/// \brief    稠密矩阵用户接口
///

#ifndef __CHIPSUM_DENSE_MATRIX_HPP__
#define __CHIPSUM_DENSE_MATRIX_HPP__


#include "impl/kokkoskernels/densemat_kokkoskernels_impl.hpp"

#include "impl/serial/densemat_serial_impl.hpp"

#include "numeric_traits.hpp"
#include "scalar.hpp"
#include "vector.hpp"


#include "../common/data_types.hpp"



namespace ChipSum {
namespace Numeric {


template <typename... Props>
class DenseMatrix {

public:
    using traits =
    DenseMatrix_Traits< Props...>;
    using matrix_type = typename traits::matrix_type;
    using matrix_type_ref =
    typename std::add_lvalue_reference<matrix_type>::type;
    using const_matrix_type_ref =
    typename std::add_const<matrix_type_ref>::type;

    using size_type = typename traits::size_type;
    using const_size_type = const size_type;
    using const_size_type_ref = const size_type&;

    using value_type = typename traits::value_type;

    using vector_type =
    ChipSum::Numeric::Vector< Props...>;

private:
    matrix_type __data;
    size_type __nrow;
    size_type __ncol;

public:

    ///
    /// \brief DenseMatrix 构造一个M行N列的稠密矩阵，该矩阵未初始化
    /// \param M M 行数
    /// \param N N 列数
    ///
    CHIPSUM_DECLARED_FUNCTION DenseMatrix(const_size_type M, const_size_type N)
        : __nrow(M), __ncol(N) {
        ChipSum::Numeric::Impl::DenseMat::create(__data,M,N);
    }


    ///
    /// \brief DenseMatrix 构造一个M行N列的稠密矩阵，并用src赋值该矩阵
    /// \param M M 行数
    /// \param N N 列数
    /// \param src 原数据
    ///
    CHIPSUM_DECLARED_FUNCTION DenseMatrix(const_size_type M, const_size_type N,
                                          value_type *src)
        : __nrow(M), __ncol(N) {

        ChipSum::Numeric::Impl::DenseMat::create(__data,M, N, src);

    }


    ///
    /// \brief GetData 获取矩阵数据
    /// \return 后端数据
    ///
    CHIPSUM_FUNCTION_INLINE matrix_type_ref GetData() {
        return __data;
    }


    ///
    /// \brief GetRowNum 获取矩阵行数
    /// \return  矩阵行数
    ///
    CHIPSUM_FUNCTION_INLINE size_type GetRowNum() { return __nrow; }


    ///
    /// \brief GetColNum 获取矩阵列数
    /// \return 矩阵列数
    ///
    CHIPSUM_FUNCTION_INLINE size_type GetColNum() { return __ncol; }


    template<typename IDT>
    ///
    /// \brief SetRow
    /// \param i 行索引
    /// \param x
    ///
    CHIPSUM_FUNCTION_INLINE void SetRow(IDT i,vector_type& x){
        ChipSum::Numeric::Impl::DenseMat::set_row(__data,x.GetData(),i);
    }

    template<typename IDT>
    ///
    /// \brief SetCol
    /// \param i 列索引
    /// \param x
    ///
    CHIPSUM_FUNCTION_INLINE void SetCol(IDT i,vector_type& x){
        ChipSum::Numeric::Impl::DenseMat::set_col(__data, x.GetData(), i);
    }

    template<typename IDT>
    ///
    /// \brief GetRowCopy 获取某一行的拷贝数据
    /// \param i 行索引
    /// \param x
    ///
    CHIPSUM_FUNCTION_INLINE void GetRowCopy(IDT i,vector_type& x){
        ChipSum::Numeric::Impl::DenseMat::get_row_copy(__data, x.GetData(), i);
    }

    template<typename IDT>
    ///
    /// \brief GetColCopy 获取某一列的拷贝数据
    /// \param i 列索引
    /// \param x
    ///
    CHIPSUM_FUNCTION_INLINE void GetColCopy(IDT i,vector_type& x){
        ChipSum::Numeric::Impl::DenseMat::get_col_copy(__data, x.GetData(), i);
    }

    template<typename IDT>
    ///
    /// \brief GetRowSlice 获取某一列的拷贝数据
    /// \param idx 行索引
    /// \param i 行终止索引
    /// \param j 行终止索引
    /// \param x
    ///
    CHIPSUM_FUNCTION_INLINE void GetRowSlice(IDT idx, IDT i, IDT j, vector_type& x){
        ChipSum::Numeric::Impl::DenseMat::get_row_slice(__data, x.GetData(), idx, i, j);
    }

    template<typename IDT>
    ///
    /// \brief GetColSlice 获取某一列的拷贝数据
    /// \param idx 列索引
    /// \param i 列起始索引
    /// \param j 列终止索引
    /// \param x
    ///
    CHIPSUM_FUNCTION_INLINE void GetColSlice(IDT idx, IDT i, IDT j, vector_type& x){
        ChipSum::Numeric::Impl::DenseMat::get_col_slice(__data, x.GetData(), idx, i, j);
    }

    template<typename IDT>
    ///
    /// \brief GetPartSlice 获取某一局部矩阵的拷贝数据
    /// \param l_i 左上角行索引
    /// \param l_j 左上角列索引
    /// \param r_i 右下角行索引
    /// \param r_j 右下角列索引
    /// \param x
    ///
    CHIPSUM_FUNCTION_INLINE void GetPartSlice(IDT l_i, IDT l_j, IDT r_i, IDT r_j, DenseMatrix& x){
        ChipSum::Numeric::Impl::DenseMat::get_part_slice(__data, x.GetData(), l_i, l_j, r_i, r_j);
    }

    ///
    /// \brief Device端到Host端数据深拷贝
    ///
    CHIPSUM_FUNCTION_INLINE void DeviceToHost(){
        ChipSum::Numeric::Impl::DenseMat::device_to_host(__data);
    }

    ///
    /// \brief Host端到Device端数据深拷贝
    ///
    CHIPSUM_FUNCTION_INLINE void HostToDevice(){
        ChipSum::Numeric::Impl::DenseMat::host_to_device(__data);
    }

    ///
    /// \brief operator * GEMM
    /// \param m 稠密矩阵
    /// \return 稠密矩阵（结果）
    ///
    CHIPSUM_FUNCTION_INLINE DenseMatrix operator*(DenseMatrix &m) {
        DenseMatrix ret(__nrow, m.GetColNum());
        ChipSum::Numeric::Impl::DenseMat::gemm(
                    __data,m.GetData(), ret.GetData());
        return ret;
    }

    template<typename ...Args>
    ///
    /// \brief GEMM C=A*B 当C为已初始化的矩阵时，强烈建议采用此接口进行GEMM运算
    /// \param B  参与运算的另一矩阵
    /// \param C  结果
    ///
    CHIPSUM_FUNCTION_INLINE void GEMM(DenseMatrix &B,DenseMatrix& C,Args... args) {
        ChipSum::Numeric::Impl::DenseMat::gemm(
                    __data,B.GetData(), C.GetData(),args...);

    }

    ///
    /// \brief LU分解，结果存入原矩阵
    /// \param tiny 分解精度,默认为0
    ///
    CHIPSUM_FUNCTION_INLINE void LU(const value_type tiny = 0) {
        ChipSum::Numeric::Impl::DenseMat::lu(__data,tiny);
    }

    ///
    /// \brief QR分解，结果存入原矩阵
    /// \param 输出矩阵系数tau
    /// \param 输出矩阵系数w
    ///
    CHIPSUM_FUNCTION_INLINE void QR(vector_type &x,vector_type& y) {
        ChipSum::Numeric::Impl::DenseMat::qr(__data,x.GetData(),y.GetData());
    }

    ///
    /// \brief HESSENBERG变换，结果存入原矩阵
    /// \param 输出矩阵系数tau
    /// \param 输出矩阵系数w
    ///
    CHIPSUM_FUNCTION_INLINE void HESSENBERG(vector_type &t,vector_type& w) {
        ChipSum::Numeric::Impl::DenseMat::hessenberg(__data,t.GetData(),w.GetData());
    }

    ///
    /// \brief op(A)*X = alpha*B if side == "L" or "l", X*op(A) = alpha*B if side == "R" or "r"
    /// \param *this  [IN,OUT]输入/输出矩阵B，输入时，多重RHS的M×N矩阵。输出时，用求解的x覆盖。
    /// \param A      [IN]矩阵A,上三角矩阵或下三角矩阵
    /// \param alpha  [IN]标量系数
    /// \param side[] [IN]op(A)*X时"L" or "l"，X*op(A)时"R" or "r"
    /// \param uplo[] [IN]A是上三角矩阵"U" or "u",下三角矩阵"L" or "l"
    /// \param trans[][IN]op(A)当"N" or "n"非转置,"T" or "t"转置, "C" or "c"伴随
    /// \param diag[] [IN]"U" or "u" 对角线参数，对角线为unit，"N" or "n"对角线非unit
    ///
    CHIPSUM_FUNCTION_INLINE void TRSM(DenseMatrix &A,
                                      const value_type alpha,
                                      const char side[],
                                      const char uplo[],
                                      const char trans[]="N",
                                      const char diag[]="N") {
        ChipSum::Numeric::Impl::DenseMat::trsm(A.GetData(),__data,alpha,side,uplo,trans,diag);
    }

    ///
    /// \brief B = alpha * op(A) * B if side == "L" or "l"  B = alpha * B * op(A) if side == "R" or "r"
    /// \param *this  [IN,OUT]输入/输出矩阵B，输入时，多重RHS的M×N矩阵。输出时，用求解的x覆盖。
    /// \param A      [IN]矩阵A,上三角矩阵或下三角矩阵
    /// \param alpha  [IN]标量系数
    /// \param side[] [IN]op(A)*X时"L" or "l"，X*op(A)时"R" or "r"
    /// \param uplo[] [IN]A是上三角矩阵"U" or "u",下三角矩阵"L" or "l"
    /// \param trans[][IN]op(A)当"N" or "n"非转置,"T" or "t"转置, "C" or "c"伴随
    /// \param diag[] [IN]"U" or "u" 对角线参数，对角线为unit，"N" or "n"对角线非unit
    ///
    CHIPSUM_FUNCTION_INLINE void TRMM(DenseMatrix &A,
                                      const value_type alpha,
                                      const char side[],
                                      const char uplo[],
                                      const char trans[]="N",
                                      const char diag[]="N") {
        ChipSum::Numeric::Impl::DenseMat::trsm(A.GetData(),__data,alpha,side,uplo,trans,diag);
    }

    ///
    /// \brief 上/下三角矩阵的逆, A = inv(A)
    /// \param *this  [IN,OUT]输入/输出矩阵A/inv(A)
    /// \param diag[] [IN]"U" or "u" 对角线参数，对角线为unit,"N" or "n"对角线非unit
    /// \return int    0成功,i如果矩阵的第i对角线元素为零，且无法完成计算
    ///
    CHIPSUM_FUNCTION_INLINE int TRTRI(const char uplo[],const char diag[]="N") {
        return ChipSum::Numeric::Impl::DenseMat::trtri(__data,uplo,diag);
    }
    

    ///
    /// \brief operator * GEMV
    /// \param v 向量
    /// \return 向量（结果）
    ///
    CHIPSUM_FUNCTION_INLINE vector_type operator*(vector_type &v) {
        vector_type ret(__nrow);


        ChipSum::Numeric::Impl::DenseMat::gemv(
                    __data, v.GetData(), ret.GetData());
        return ret;
    }

    template<typename ...Args>
    ///
    /// \brief GEMM C=A*B 当C为已初始化的矩阵时，强烈建议采用此接口进行GEMM运算
    /// \param B  参与运算的另一矩阵
    /// \param C  结果
    ///
    CHIPSUM_FUNCTION_INLINE void GEMV(vector_type &x,vector_type& y,Args... args) {
        ChipSum::Numeric::Impl::DenseMat::gemv(
                    __data,x.GetData(), y.GetData(),args...);

    }

    ///
    /// \brief operator *= A*=a
    /// \param a 系数
    /// \return A（结果）
    ///
    CHIPSUM_FUNCTION_INLINE DenseMatrix operator*(const value_type& a) {
        DenseMatrix<Props...> ret(__nrow,__ncol);
        ChipSum::Numeric::Impl::DenseMat::scal( __data,ret.GetData(),a);
        return ret;
    }

    ///
    /// \brief operator *= A*=a
    /// \param a 系数
    /// \return A（结果）
    ///
    CHIPSUM_FUNCTION_INLINE DenseMatrix& operator*=(const value_type& a) {
        ChipSum::Numeric::Impl::DenseMat::scal( __data,__data,a);
        return *this;
    }

    ///
    /// \brief operator /= A/=a
    /// \attention 后续希望将1/a变为类似ChipSum::Numeric::Const<ValueType>::one()/a;
    /// \param a 系数
    /// \return A（结果）
    ///
    CHIPSUM_FUNCTION_INLINE DenseMatrix& operator/=(const value_type& a) {
        value_type one = static_cast<value_type>(1);
        ChipSum::Numeric::Impl::DenseMat::scal( __data,__data,(one/a));
        return *this;
    }


    ///
    /// \brief operator () 获取A(i,j)
    /// \param i 行索引
    /// \param j 列索引
    /// \return A(i,j)
    ///
    CHIPSUM_FUNCTION_INLINE value_type &operator()(const_size_type_ref i,
                                                   const_size_type_ref j) {
        return ChipSum::Numeric::Impl::DenseMat::get_item(
                    __data,i,j);
    }



    ///
    /// \brief operator () 获取A(i,j)（只读）
    /// \param i 行数
    /// \param j 列数
    /// \return A(i,j)
    ///
    CHIPSUM_FUNCTION_INLINE const value_type &operator()(const_size_type_ref i,
                                                         const_size_type_ref j) const{
        return ChipSum::Numeric::Impl::DenseMat::get_item(
                    __data,i,j);
    }


    ///
    /// \brief operator [], 获取device端 A(i,j)值
    ///        device端 仅返回二维中的该行的首地址，列由C++自身寻址完成
    /// \param i 行索引
    /// \param j 列索引
    /// \return A[i,0]
    ///
    CHIPSUM_SPECIAL_INLINE value_type *operator[](const_size_type_ref i) const{
        return ChipSum::Numeric::Impl::DenseMat::item(__data, i, 0);
    }

    ///
    /// \brief Item函数, 获取device端 A(i,j)值
    /// \param i 行索引
    /// \param j 列索引
    /// \return A[i,j]
    ///
    CHIPSUM_SPECIAL_INLINE value_type & Item(const_size_type_ref i,
                                              const_size_type_ref j) const{
        return *(ChipSum::Numeric::Impl::DenseMat::item(__data, i, j));
    }

    // ********************** AI op start ********************** //
    
    ///
    /// \brief dense, without bias, out = __data * weight
    /// \return out
    /// 
    CHIPSUM_FUNCTION_INLINE void Dense(DenseMatrix& weight, DenseMatrix& out) {
        return ChipSum::Numeric::Impl::DenseMat::dense(__data, weight.GetData(), out.GetData());
    }

    ///
    /// \brief dense, without bias, out = __data * weight, with transpose parameter
    /// \return out
    /// 
    CHIPSUM_FUNCTION_INLINE void Dense(const char transA[], const char transB[], 
                                        DenseMatrix& weight, DenseMatrix& out) {
        return ChipSum::Numeric::Impl::DenseMat::dense(__data, weight.GetData(), out.GetData(), transA, transB);
    }

    ///
    /// \brief dense, with bias, out = __data * weight + bias
    /// \return out
    /// 
    CHIPSUM_FUNCTION_INLINE void Dense(DenseMatrix& weight, vector_type& bias, DenseMatrix& out) {
        return ChipSum::Numeric::Impl::DenseMat::dense(__data, weight.GetData(), bias.GetData(), out.GetData());
    }

    ///
    /// \brief dense, with bias, out = __data * weight + bias, with transpose parameter
    /// \return out
    /// 
    CHIPSUM_FUNCTION_INLINE void Dense(const char transA[], const char transB[], 
                                        DenseMatrix& weight, vector_type& bias, DenseMatrix& out) {
        return ChipSum::Numeric::Impl::DenseMat::dense(__data, weight.GetData(), bias.GetData(), out.GetData(), transA, transB);
    }

    ///
    /// \brief Relu, val>0 ? val : 0
    /// \return A[i,j]
    /// 
    CHIPSUM_FUNCTION_INLINE void Relu() {
        return ChipSum::Numeric::Impl::DenseMat::relu(__data);
    }

    ///
    /// \brief LeakyRelu, val>0 ? val : 0.01*val
    /// \return A[i,j]
    /// 
    CHIPSUM_FUNCTION_INLINE void LeakyRelu() {
        return ChipSum::Numeric::Impl::DenseMat::leakyrelu(__data);
    }

    ///
    /// \brief Softmax
    /// \return A[i,j]
    /// 
    CHIPSUM_FUNCTION_INLINE void Softmax() {
        return ChipSum::Numeric::Impl::DenseMat::softmax(__data);
    }

    ///
    /// \brief logSoftmax
    /// \return A[i,j]
    /// 
    CHIPSUM_FUNCTION_INLINE void LogSoftmax() {
        return ChipSum::Numeric::Impl::DenseMat::softmax(__data, true);
    }

    ///
    /// \brief norm
    /// \return A[i,j]
    /// 
    CHIPSUM_FUNCTION_INLINE void Norm() {
        return ChipSum::Numeric::Impl::DenseMat::norm(__data);
    }

    ///
    /// \brief argma
    /// \return int index of the max value
    /// 
    template<typename OStreamT=std::ostream>
    CHIPSUM_FUNCTION_INLINE void Argmax(OStreamT &out = std::cout) {
        ChipSum::Numeric::Impl::DenseMat::argmax(__data,out);
    }

    // ********************** AI op finish ********************** //

    template<typename OStreamT=std::ostream>
    ///
    /// \brief Print 打印函数
    /// \param out 输出流
    ///
    CHIPSUM_FUNCTION_INLINE void Print(OStreamT &out = std::cout) {
        ChipSum::Numeric::Impl::DenseMat::print(
                    __data,out);
    }
};

} // End namespace Numeric
} // End namespace ChipSum

///
/// \brief Matrix 默认的
///
typedef ChipSum::Numeric::DenseMatrix<CSFloat,ChipSum::Backend::DefaultBackend>
CSMatrix;

typedef ChipSum::Numeric::DenseMatrix<CSFloat,ChipSum::Backend::Serial>
SerialMatrix;

#endif // __CHIPSUM_DENSE_MATRIX_HPP__
