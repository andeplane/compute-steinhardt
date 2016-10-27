/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(steinhardt/atom,ComputeSteinhardtAtom)

#else

#ifndef LMP_COMPUTE_STEINHARDT_ATOM_H
#define LMP_COMPUTE_STEINHARDT_ATOM_H

#include "compute.h"
#include <complex>

namespace LAMMPS_NS {

class ComputeSteinhardtAtom : public Compute {
 public:
  ComputeSteinhardtAtom(class LAMMPS *, int, char **);
  ~ComputeSteinhardtAtom();
  void init();
  void init_list(int, class NeighList *);
  void compute_peratom();
  double memory_usage();

 private:
  int nmax,maxneigh,ncol,nnn;
  int l; // which l in q_l do we want?
  class NeighList *list;
  double *distsq;
  int *nearest;
  double **rlist;
  double qMin, qMax;
  int qmax;
  double **qnarray;
  double cutsq;
  std::complex<double> **qnm; // To be removed in non-boost version
  double **qnm_r;
  double **qnm_i;
  int    **nearestNeighborList;

  void select3(int, int, double *, int *, double **);
  void calc_boop(int atomIndex);
  void calc_boop_boost_complex(int atomIndex); // To be removed in non-boost version
  double dist(const double r[]);

  double spherical_harmonic_without_polar_angle(int l, int m, double phi, double cosTheta);
  //double associated_legendre(int l, int m, double x);
  double renormalized_legendre_positive_m(int l, int m, double x);
  std::pair<double, double> spherical_harmonic(int l, int m, double phi, double cosTheta);

};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute orientorder/atom requires a pair style be defined

Self-explantory.

E: Compute orientorder/atom cutoff is longer than pairwise cutoff

Cannot compute order parameter beyond cutoff.

W: More than one compute orientorder/atom

It is not efficient to use compute orientorder/atom more than once.

*/
