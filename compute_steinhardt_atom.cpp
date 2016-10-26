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
   Contributing author:  Aidan Thompson (SNL)
                         Axel Kohlmeyer (Temple U)
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
    distsq(NULL), nearest(NULL), rlist(NULL), qnarray(NULL), qnm_r(NULL), qnm_i(NULL), nearestNeighborList(NULL)
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
    memory->destroy(qnarray);
    memory->destroy(distsq);
    memory->destroy(rlist);
    memory->destroy(nearest);
    memory->destroy(qnm_r);
    memory->destroy(qnm_i);
    memory->destroy(vector_atom);
    
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
        memory->destroy(qnm_r);
        memory->destroy(qnm_i);
        memory->destroy(nearestNeighborList);
        memory->destroy(vector_atom);
        nmax = atom->nmax;
        memory->create(qnarray,nmax,ncol,"orientorder/atom:qnarray");
        memory->create(qnm_r,nmax,2*l+1,"orientorder/atom:qnm_r");
        memory->create(qnm_i,nmax,2*l+1,"orientorder/atom:qnm_i");
        memory->create(nearestNeighborList,nmax,nnn,"orientorder/atom:nearestNeighborList");
        memory->create(vector_atom,nmax,"orientorder/atom:vector_atom");
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
        qnarray[i][0] = -2;
        for(j=1; j<ncol; j++) {
            qnarray[i][j] = -2;
        }

        vector_atom[i] = -2; // Reset compute values for this atom
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
                qnm_r[i][mIndex] = -2;
                qnm_i[i][mIndex] = -2;
            }
            
            if ((ncount == 0) || (ncount < nnn)) {
                continue;
            }
            
            // if nnn > 0, use only nearest nnn neighbors
            
            if (nnn > 0) {
                select3(nnn,ncount,distsq,nearest,rlist);
                ncount = nnn;
            }
            
            for(int j=0; j<nnn; j++) {
                nearestNeighborList[i][j] = nearest[j];
            }
            
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
              double Qij_real = 0;
              double Qij_imag = 0;
              double iSum = 0; // normalization for i atom
              double jSum = 0; // normalization for j atom
              for(int mIndex=0; mIndex<2*l+1; mIndex++) {
                double a = qnm_r[i][mIndex]; // q_l for atom i is a+ib
                double b = qnm_i[i][mIndex];

                double c = qnm_r[j][mIndex]; // q_l for atom j is c+id
                double d = qnm_i[j][mIndex];
                
                Qij_real += a*c + b*d; // (a+ib) * (c + id) real part
                Qij_imag += a*d - b*c; // (a+ib) * (c + id) imaginary part
                
                iSum += a*a + b*b; // |a+ib|^2
                jSum += c*c + d*d; // |c+id|^2
              }
              
              Qij_real /= sqrt(iSum*jSum); // normalize by the product of the normalization factors
              // cout << "Q_ij_real = " << Qij_real << " and Q_ij_imag = " << Qij_imag / sqrt(iSum*jSum) << endl;
              qnarray[i][jj+1] = Qij_real;

              if(qMin <= Qij_real && Qij_real <= qMax) {  
                qnarray[i][0] += 1;
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
    // double bytes = ncol*nmax * sizeof(double);
    // bytes += (qmax*(2*qmax+1)+maxneigh*4) * sizeof(double); 
    // bytes += (nqlist+maxneigh) * sizeof(int); 
    // return bytes;
  return 0;
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

void ComputeSteinhardtAtom::calc_boop(int atomIndex) {
    for(int ineigh = 0; ineigh < nnn; ineigh++) {
        const double * const r = rlist[ineigh];
        double rmag = dist(r);
        if(rmag <= MY_EPSILON) {
            return;
        }
        
        double costheta = r[2] / rmag;
        double phi = atan2(r[1], r[0]);
        for(int mIndex = 0; mIndex < 2*l+1; mIndex++) {
            int m = mIndex - l; // from -l to +l
            double cosMPhi = cos(m*phi);
            double sinMPhi = sin(m*phi);
            double prefactor = polar_prefactor(l, m, costheta);
            qnm_r[atomIndex][mIndex] = prefactor*cosMPhi;
            qnm_i[atomIndex][mIndex] = prefactor*sinMPhi;
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
   polar prefactor for spherical harmonic Y_l^m, where 
   Y_l^m (theta, phi) = prefactor(l, m, cos(theta)) * exp(i*m*phi)
------------------------------------------------------------------------- */

double ComputeSteinhardtAtom::
polar_prefactor(int l, int m, double costheta) {
    const int mabs = abs(m);
    
    double prefactor = 1.0;
    for (int i=l-mabs+1; i < l+mabs+1; ++i)
        prefactor *= static_cast<double>(i);
    
    prefactor = sqrt(static_cast<double>(2*l+1)/(MY_4PI*prefactor))
            * associated_legendre(l,mabs,costheta);
    
    if ((m >= 0) && (m % 2)) prefactor = -prefactor;
    
    return prefactor;
}

/* ----------------------------------------------------------------------
   associated legendre polynomial
------------------------------------------------------------------------- */

double ComputeSteinhardtAtom::
associated_legendre(int l, int m, double x) {
    if (l < m) return 0.0;
    
    double p(1.0), pm1(0.0), pm2(0.0);
    
    if (m != 0) {
        const double sqx = sqrt(1.0-x*x);
        for (int i=1; i < m+1; ++i)
            p *= static_cast<double>(2*i-1) * sqx;
    }
    
    for (int i=m+1; i < l+1; ++i) {
        pm2 = pm1;
        pm1 = p;
        p = (static_cast<double>(2*i-1)*x*pm1
             - static_cast<double>(i+m-1)*pm2) / static_cast<double>(i-m);
    }
    
    return p;
}

