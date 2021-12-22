///
/// \file     ex1.cpp
/// \author   yaojie yu
/// \group    CDCS-HPC
/// \date     2021-12-14
/// \brief    %stuff%
///





#include "ChipSum.hpp"
#include "chipsum/chipsum_macro.h"

Vector cg(CSR &A, Vector &b, Vector &x, double tol, int max_it)
{
    // Conjugate Gradient Method without preconditioning.
    //
    // input   A        REAL matrix
    //         x        REAL initial guess vector
    //         b        REAL right hand side vector
    //         tol      REAL error tolerance
    //         max_it   INTEGER maximum number of iterations
    //
    // output  x        REAL solution vector


    Vector r(x.GetSize());
    A.SPMV(x, r);
    b.AXPBY(r, 1.0, -1.0); //r = b - A*x

    Vector p(b.GetSize());
    p.DeepCopy(r);

    Vector Ap(b.GetSize());

    double alpha = 0, beta = 0.0, rsnew = 0;
    double rsold = r.Dot(r);

    for (int i = 0; i < max_it; i++)
    {

        A.SPMV(p, Ap);

        alpha = rsold / (p.Dot(Ap));

        p.AXPBY(x, alpha, 1.0);

        Ap.AXPBY(r, -alpha, 1.0);

        rsnew = r.Dot(r);

        if (sqrt(rsnew) < tol)
            break;

        beta = rsnew / rsold;
        r.AXPBY(p, 1.0, beta);


        rsold = rsnew;



    }

    return x;
}



int main(int argc, char *argv[])
{

    /* .mtx格式数据，如$HOME/ChipSum/data/A.mtx */
    char* filename_A = argv[1];
    
    /* .mtx格式数据，如$HOME/ChipSum/data/b.csv */
    char* filename_b = argv[2];


    ChipSum::Common::Init(argc, argv);
    {

        CSInt nv = 0, ne = 0;
        CSInt *xadj, *adj;
        double *ew;

        KokkosKernels::Impl::read_matrix<CSInt,CSInt, double> (&nv, &ne, &xadj, &adj, &ew, filename_A);

        CSR A(nv,nv,ne,xadj,adj,ew);




        vector<double> b_data;
        double temp;

        ifstream IN(filename_b);

        for(int i=0;i<nv;++i){
            temp = 1.;
            b_data.push_back(temp);
        }


        IN.close();

        Vector b(nv,b_data.data());

        Vector x0(nv);

        x0*=0;


        double tol = 1e-12;
        int max_it = 500;

        auto sol_cg = cg(A, b, x0, tol, max_it);

        delete xadj;
        delete adj;
        delete ew;
    }
    ChipSum::Common::Finalize();
}
