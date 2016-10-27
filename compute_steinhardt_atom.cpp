/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author:  Anders Hafreager (University of Oslo)
                        Henrik Andersen Sveinsson (University of Oslo)
    Modification of compute_steinhardt_atom by ??
------------------------------------------------------------------------- */

#include <string.h>
#include <stdlib.h>
#include "compute_steinhardt_atom.h"
#include "atom.h"
#include "update.h"
#include "modify.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"
#include <iostream>
#include <boost/math/special_functions/spherical_harmonic.hpp>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;

#ifdef DBL_EPSILON
#define MY_EPSILON (10.0*DBL_EPSILON)
#else
#define MY_EPSILON (10.0*2.220446049250313e-16)
#endif

/* ---------------------------------------------------------------------- */

ComputeSteinhardtAtom::ComputeSteinhardtAtom(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg),
    distsq(NULL), nearest(NULL), rlist(NULL), qnarray(NULL), qnm_r(NULL), qnm_i(NULL), qnm(NULL), nearestNeighborList(NULL)
{
    if (narg < 3 ) error->all(FLERR,"Illegal compute steinhardt/atom command");

    // set default values for optional args

    nnn = 12;
    cutsq = 0.0;
    qMin = 0.812;
    qMax = 1.0;
    // process optional args

    int iarg = 3;
    while (iarg < narg) {
        if (strcmp(arg[iarg],"nnn") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal compute steinhardt/atom command");
            if (strcmp(arg[iarg+1],"NULL") == 0)
                nnn = 0;
            else {
                nnn = force->numeric(FLERR,arg[iarg+1]);
                if (nnn <= 0)
                    error->all(FLERR,"Illegal compute steinhardt/atom command");
            }
            iarg += 2;
        } else if (strcmp(arg[iarg],"l") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal compute steinhardt/atom command");
            l = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        } else if (strcmp(arg[iarg],"min") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal compute steinhardt/atom command");
            qMin = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        } else if (strcmp(arg[iarg],"max") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal compute steinhardt/atom command");
            qMax = force->numeric(FLERR,arg[iarg+1]);
            iarg += 2;
        } else if (strcmp(arg[iarg],"cutoff") == 0) {
            if (iarg+2 > narg) error->all(FLERR,"Illegal compute steinhardt/atom command");
            double cutoff = force->numeric(FLERR,arg[iarg+1]);
            if (cutoff <= 0.0) error->all(FLERR,"Illegal compute steinhardt/atom command");
            cutsq = cutoff*cutoff;
            iarg += 2;
        } else error->all(FLERR,"Illegal compute steinhardt/atom command");
    }
    ncol = nnn+1;
    peratom_flag = 1;
    size_peratom_cols = ncol;

    nmax = 0;
    maxneigh = 0;
}

/* ---------------------------------------------------------------------- */

ComputeSteinhardtAtom::~ComputeSteinhardtAtom()
{
    memory->destroy(nearestNeighborList);
    memory->destroy(qnarray);
    memory->destroy(distsq);
    memory->destroy(rlist);
    memory->destroy(nearest);
    memory->destroy(qnm);
    memory->destroy(qnm_r);
    memory->destroy(qnm_i);
}

/* ---------------------------------------------------------------------- */

void ComputeSteinhardtAtom::init()
{
    if (force->pair == NULL)
        error->all(FLERR,"Compute steinhardt/atom requires a pair style be defined");
    if (cutsq == 0.0) cutsq = force->pair->cutforce * force->pair->cutforce;
    else if (sqrt(cutsq) > force->pair->cutforce)
        error->all(FLERR,
                   "Compute steinhardt/atom cutoff is longer than pairwise cutoff");

    // need an occasional full neighbor list

    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->compute = 1;
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
    neighbor->requests[irequest]->occasional = 1;

    int count = 0;
    for (int i = 0; i < modify->ncompute; i++)
        if (strcmp(modify->compute[i]->style,"steinhardt/atom") == 0) count++;
    if (count > 1 && comm->me == 0)
        error->warning(FLERR,"More than one compute steinhardt/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeSteinhardtAtom::init_list(int id, NeighList *ptr)
{
    list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeSteinhardtAtom::compute_peratom()
{
    int i,j,ii,jj,inum,jnum;
    double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
    int *ilist,*jlist,*numneigh,**firstneigh;

    invoked_peratom = update->ntimestep;

    // grow order parameter array if necessary

    if (atom->nmax > nmax) {
        memory->destroy(qnarray);
        memory->destroy(qnm);
        memory->destroy(qnm_r);
        memory->destroy(qnm_i);
        memory->destroy(nearestNeighborList);
        nmax = atom->nmax;
        memory->create(qnarray,nmax,ncol,"orientorder/atom:qnarray");
        memory->create(qnm, nmax, 2*l+1, "orientorder/atom:qnm");
        memory->create(qnm_r,nmax,2*l+1,"orientorder/atom:qnm_r");
        memory->create(qnm_i,nmax,2*l+1,"orientorder/atom:qnm_i");
        memory->create(nearestNeighborList,nmax,nnn,"orientorder/atom:nearestNeighborList");
        array_atom = qnarray;
    }

    // invoke full neighbor list (will copy or build if necessary)

    neighbor->build_one(list);

    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;

    // compute order parameter for each atom in group
    // use full neighbor list to count atoms less than cutoff

    double **x = atom->x;
    int *mask = atom->mask;

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        qnarray[i][0] = 0;
        for(j=1; j<ncol; j++) {
            qnarray[i][j] = 0;
        }

        if (mask[i] & groupbit) { // Make sure atom i is in the group
            xtmp = x[i][0];
            ytmp = x[i][1];
            ztmp = x[i][2];
            jlist = firstneigh[i];
            jnum = numneigh[i];

            if (jnum > maxneigh) { // Reallocate if arrays too small
                memory->destroy(distsq);
                memory->destroy(rlist);
                memory->destroy(nearest);
                maxneigh = jnum;
                memory->create(distsq,maxneigh,"steinhardt/atom:distsq");
                memory->create(rlist,maxneigh,3,"steinhardt/atom:rlist");
                memory->create(nearest,maxneigh,"steinhardt/atom:nearest");
            }

            // loop over list of all neighbors within force cutoff
            // distsq[] = distance sq to each
            // rlist[] = distance vector to each
            // nearest[] = atom indices of neighbors

            int ncount = 0;
            for (jj = 0; jj < jnum; jj++) {
                j = jlist[jj];
                j &= NEIGHMASK;

                if (mask[j] & groupbit) { // Make sure atom j is in the group
                    delx = xtmp - x[j][0];
                    dely = ytmp - x[j][1];
                    delz = ztmp - x[j][2];
                    rsq = delx*delx + dely*dely + delz*delz;
                    if (rsq < cutsq) {
                        distsq[ncount] = rsq;
                        rlist[ncount][0] = delx;
                        rlist[ncount][1] = dely;
                        rlist[ncount][2] = delz;
                        nearest[ncount++] = j;
                    }
                }
            }

            // if not nnn neighbors, order parameter = nan;
            for(int mIndex=0; mIndex<2*l+1; mIndex++) {
                qnm[i][mIndex] = sqrt(-1) + 0i;
                qnm_r[i][mIndex] = sqrt(-1);
                qnm_i[i][mIndex] = sqrt(-1);
            }

            if ((ncount == 0) || (ncount < nnn)) {
                continue;
            }

            for(int mIndex=0; mIndex<2*l+1; mIndex++) {
                qnm[i][mIndex] = 0 + 0i;
                qnm_r[i][mIndex] = 0;
                qnm_i[i][mIndex] = 0;
            }

            // if nnn > 0, use only nearest nnn neighbors

            if (nnn > 0) {
                select3(nnn,ncount,distsq,nearest,rlist);
                ncount = nnn;
            }

            for(int j=0; j<nnn; j++) {
                nearestNeighborList[i][j] = nearest[j];
            }

            calc_boop_boost_complex(i);
            calc_boop(i);

        }
    }

    for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (mask[i] & groupbit) {
            jlist = nearestNeighborList[i];
            jnum = nnn;
            for (jj = 0; jj < jnum; jj++) {
              int j = jlist[jj];
              std::complex<double> Qij = 0 + 0i;
              std::complex<double> iSumc = 0 + 0i;
              std::complex<double> jSumc = 0 + 0i;
              double Qij_real = 0;
              double Qij_imag = 0;
              double iSum = 0; // normalization for i atom
              double jSum = 0; // normalization for j atom
              for(int mIndex=0; mIndex<2*l+1; mIndex++) {
                Qij += qnm[i][mIndex]*std::conj(qnm[j][mIndex]);
                iSumc += qnm[i][mIndex]*std::conj(qnm[i][mIndex]);
                jSumc += qnm[j][mIndex]*std::conj(qnm[j][mIndex]);
                double a = qnm_r[i][mIndex]; // q_l for atom i is a+ib
                double b = qnm_i[i][mIndex];

                double c = qnm_r[j][mIndex]; // q_l for atom j is c+id
                double d = qnm_i[j][mIndex];

                Qij_real += a*c + b*d; // (a+ib) * (c + id) real part
                Qij_imag += -a*d + b*c; // (a+ib) * (c + id) imaginary part

                iSum += a*a + b*b; // |a+ib|^2
                jSum += c*c + d*d; // |c+id|^2

                cout << qnm[i][mIndex].real()-a << " " << qnm[i][mIndex].imag()-b << " " << qnm[j][mIndex].real()-c << " " << qnm[j][mIndex].imag()-d << endl <<endl ;

                //cout << a << " " << b << " " << c << " " << d << endl;
                //cout << qnm[i][mIndex].real() << " " << qnm[i][mIndex].imag() << " " << qnm[j][mIndex].real() << " " << qnm[j][mIndex].imag() << endl <<endl ;

              }

              Qij_real /= sqrt(iSum*jSum); // normalize by the product of the normalization factors
              //cout << "Q_ij_real = " << Qij_real << " and Q_ij_imag = " << Qij_imag / sqrt(iSum*jSum) << endl;
              Qij = Qij/sqrt(real(iSumc)*real(jSumc));

              //cout << abs(Qij_real) - abs(std::real(Qij)) << " ";

              qnarray[i][jj+1] = Qij_real;
              //qnarray[i][jj+1] = std::real(Qij);
              //cout << Qij << endl;
              if(qMin <= std::real(Qij) && std::real(Qij) <= qMax) {
                if (qnarray[i][0] < 0)
                {
                    qnarray[i][0] = 1;
                }
                else
                {
                  qnarray[i][0] += 1;
                }
              }
            }
        }
    }

    // for (ii = 0; ii < inum; ii++) {
    //     i = ilist[ii];
    //     if (mask[i] & groupbit) {
    //         if(vector_atom[i] != -2) cout << "c[" << i << "] = " << vector_atom[i] << endl;
    //     }
    // }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeSteinhardtAtom::memory_usage()
{
    double bytes = ncol*nmax * sizeof(double); // qnarray
    bytes += nmax*(2*l+1) * sizeof(double); // qnm_r
    bytes += nmax*(2*l+1) * sizeof(double); // qnm_i
    bytes += nmax*nnn * sizeof(int); // nearestNeighborList
    bytes += maxneigh * sizeof(double); // distsq
    bytes += 3*maxneigh * sizeof(double); // rlist
    bytes += maxneigh * sizeof(int); // nearest
    return bytes;
}

/* ----------------------------------------------------------------------
   select3 routine from Numerical Recipes (slightly modified)
   find k smallest values in array of length n
   sort auxiliary arrays at same time
------------------------------------------------------------------------- */

// Use no-op do while to create single statement

#define SWAP(a,b) do {        \
    tmp = a; a = b; b = tmp;      \
    } while(0)

#define ISWAP(a,b) do {       \
    itmp = a; a = b; b = itmp;      \
    } while(0)

#define SWAP3(a,b) do {       \
    tmp = a[0]; a[0] = b[0]; b[0] = tmp;  \
    tmp = a[1]; a[1] = b[1]; b[1] = tmp;  \
    tmp = a[2]; a[2] = b[2]; b[2] = tmp;  \
    } while(0)

/* ---------------------------------------------------------------------- */

void ComputeSteinhardtAtom::select3(int k, int n, double *arr, int *iarr, double **arr3)
{
    int i,ir,j,l,mid,ia,itmp;
    double a,tmp,a3[3];

    arr--;
    iarr--;
    arr3--;
    l = 1;
    ir = n;
    for (;;) {
        if (ir <= l+1) {
            if (ir == l+1 && arr[ir] < arr[l]) {
                SWAP(arr[l],arr[ir]);
                ISWAP(iarr[l],iarr[ir]);
                SWAP3(arr3[l],arr3[ir]);
            }
            return;
        } else {
            mid=(l+ir) >> 1;
            SWAP(arr[mid],arr[l+1]);
            ISWAP(iarr[mid],iarr[l+1]);
            SWAP3(arr3[mid],arr3[l+1]);
            if (arr[l] > arr[ir]) {
                SWAP(arr[l],arr[ir]);
                ISWAP(iarr[l],iarr[ir]);
                SWAP3(arr3[l],arr3[ir]);
            }
            if (arr[l+1] > arr[ir]) {
                SWAP(arr[l+1],arr[ir]);
                ISWAP(iarr[l+1],iarr[ir]);
                SWAP3(arr3[l+1],arr3[ir]);
            }
            if (arr[l] > arr[l+1]) {
                SWAP(arr[l],arr[l+1]);
                ISWAP(iarr[l],iarr[l+1]);
                SWAP3(arr3[l],arr3[l+1]);
            }
            i = l+1;
            j = ir;
            a = arr[l+1];
            ia = iarr[l+1];
            a3[0] = arr3[l+1][0];
            a3[1] = arr3[l+1][1];
            a3[2] = arr3[l+1][2];
            for (;;) {
                do i++; while (arr[i] < a);
                do j--; while (arr[j] > a);
                if (j < i) break;
                SWAP(arr[i],arr[j]);
                ISWAP(iarr[i],iarr[j]);
                SWAP3(arr3[i],arr3[j]);
            }
            arr[l+1] = arr[j];
            arr[j] = a;
            iarr[l+1] = iarr[j];
            iarr[j] = ia;
            arr3[l+1][0] = arr3[j][0];
            arr3[l+1][1] = arr3[j][1];
            arr3[l+1][2] = arr3[j][2];
            arr3[j][0] = a3[0];
            arr3[j][1] = a3[1];
            arr3[j][2] = a3[2];
            if (j >= k) ir = j-1;
            if (j <= k) l = i;
        }
    }
}


/* ----------------------------------------------------------------------
   calculate the bond orientational order parameters
------------------------------------------------------------------------- */
void ComputeSteinhardtAtom::calc_boop(int atomIndex)
{
    for(int ineigh = 0; ineigh < nnn; ineigh++) {
        const double * const r = rlist[ineigh];
        double rdist = dist(r);
        double cosTheta = r[2]/rdist;
        //double theta = atan2(sqrt(r[0]*r[0]+r[1]*r[1]), r[2]);
        double phi = atan2(r[1], r[0]);
        for(int mIndex = 0; mIndex < 2*l+1; mIndex++) {
            int m = mIndex - l; // from -l to +l
            std::pair<double, double> sph_harm = spherical_harmonic(l, m, phi, cosTheta);
            qnm_r[atomIndex][mIndex] += sph_harm.first;
            qnm_i[atomIndex][mIndex] += sph_harm.second;
        }
    }
}

void ComputeSteinhardtAtom::calc_boop_boost_complex(int atomIndex) {
    for(int ineigh = 0; ineigh < nnn; ineigh++) {
        const double * const r = rlist[ineigh];
        double theta = atan2(sqrt(r[0]*r[0]+r[1]*r[1]), r[2]);
        double phi = atan2(r[1], r[0]);
        for(int mIndex = 0; mIndex < 2*l+1; mIndex++) {
            int m = mIndex - l; // from -l to +l
            qnm[atomIndex][mIndex] += boost::math::spherical_harmonic(l, m, theta, phi);
        }
    }
}

/* ----------------------------------------------------------------------
   calculate scalar distance
------------------------------------------------------------------------- */

double ComputeSteinhardtAtom::dist(const double r[]) {
    return sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
}


/* ----------------------------------------------------------------------
   associated legendre polynomial
------------------------------------------------------------------------- */
/*
double associated_legendre(int l, int m, double x)
{
    const double PI = 3.14159265358979323;
    if (m<0 || m>l || abs(x) > 1.0)
        throw("Bad arguments in routine associated_legendre");
    double prod = 1.0;
    for (int j=l-m+1; j<=l+m; j++)
        prod *= j;
    return sqrt(4.0*PI*prod/(2*l+1))*renormalized_legendre(l,m,x)
}
*/

/* -----------------------------------------------------------
    Computes the renormalized associated legendre polynomial

    sqrt((2l+1)/4pi*(l-m)!/(l+m)!)P_l^m
    P_l^m (x) = (-1)^m(1-x^2)^{m/2} d^m/dx^m P_l(x)
    P_0(x) = 1

-------------------------------------------------------------- */

std::pair<double, double> ComputeSteinhardtAtom::spherical_harmonic(int l, int m, double phi, double cosTheta)
{
    double real_part = renormalized_legendre_positive_m(l, abs(m), cosTheta);
    std::pair<double, double> result;
    result.first = real_part*cos(m*phi);
    result.second = real_part*sin(m*phi);
    if (m>=0)
    {
        return result;
    }
    else
    {
        result.first = result.first*pow(-1, abs(m));
        result.second  = result.second*pow(-1, abs(m));
        return result;
    }
}

double ComputeSteinhardtAtom::renormalized_legendre_positive_m(int l, int m, double x)
{
    static const double PI = 3.14159265358979323;
    int i, ll;
    double fact, oldfact, pll, pmm, pmmp1, omx2;
    if ( m<0 || m>l || abs(x) > 1.0 )
        throw("Bad arguments in routine renormalized_legendre");
    pmm = 1.0;
    if (m>0)
    {
        omx2 = (1.0-x)*(1.0+x);
        fact=1.0;
        for (i=1; i<=m; i++)
        {
            pmm *= omx2*fact/(fact+1.0);
            fact += 2.0;
        }
    }
    pmm = sqrt((2*m+1)*pmm/(4.0*PI));
    if (m & 1)
        pmm = -pmm;
    if (l == m)
        return pmm;
    else
    {
        pmmp1=x*sqrt(2.0*m+3.0)*pmm;
        if (l == (m+1))
            return pmmp1;
        else
        {
            oldfact = sqrt(2.0*m+3.0);
            for (ll=m+2;ll<=l;ll++)
            {
                fact = sqrt((4.0*ll*ll-1.0)/(ll*ll-m*m));
                pll = (x*pmmp1-pmm/oldfact)*fact;
                oldfact = fact;
                pmm = pmmp1;
                pmmp1=pll;
            }
            return pll;
        }
    }
}
