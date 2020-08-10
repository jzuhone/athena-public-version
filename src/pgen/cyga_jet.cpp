//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file cyga_jet.cpp
//  \brief Problem generator for simple simulation of Cygnus A jet hitting ICM boundary

// C headers

// C++ headers
#include <cmath>      // sqrt()
#include <cstring>    // strcmp()
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../athena.hpp"                   // macros, enums
#include "../athena_arrays.hpp"            // AthenaArray
#include "../coordinates/coordinates.hpp"  // Coordinates
#include "../eos/eos.hpp"                  // EquationOfState
#include "../field/field.hpp"              // Field
#include "../hydro/hydro.hpp"              // Hydro
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"          // ParameterInput

// Configuration checking
#if not RELATIVISTIC_DYNAMICS
#error "This problem generator must be used with relativity"
#endif

namespace {
    Real d_amb, p_amb, vx_amb, vy_amb, vz_amb, bx_amb, by_amb, bz_amb;
    Real r_jet, d_jet, p_jet, vx_jet, vy_jet, vz_jet, bx_jet, by_jet, bz_jet;
    Real gm1, x_jet, x3_0, icm_pos0, icm_angle, rho_ratio, rho_left, rho_right;
    Real pgas_left, pgas_right, vx_left, vx_right, vy_left, vy_right;
    Real vz_left, vz_right, bbx_left, bbx_right, bby_left, bby_right;
    Real bbz_left, bbz_right;
} // namespace

// BCs on L-x1 (left edge) of grid with jet inflow conditions
void JetInnerY1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh);

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
    // initialize global variables

    // Fluid parameters
    icm_pos0   = pin->GetReal("problem", "icm_pos0");
    icm_angle  = pin->GetReal("problem", "icm_angle");
    icm_angle *= PI/180;
    tan_icm    = std::tan(icm_angle);

    std::stringstream msg;

    if (icm_pos0 < pmy_mesh->mesh_size.x1min || icm_pos0 > pmy_mesh->mesh_size.x1max) {
        msg << "### FATAL ERROR in Problem Generator\n"
            << "icm_pos0=" << icm_pos0 << " lies outside domain" << std::endl;
        ATHENA_ERROR(msg);
    }

    // left side of interface
    Real rho_left = pin->GetReal("problem", "dl");
    Real pgas_left = pin->GetReal("problem", "pl");
    Real rho_ratio = pin->GetReal("problem", "dratio");
    Real bbx_left = 0.0, bby_left = 0.0, bbz_left = 0.0;
    Real vx_left = 0.0, vy_left = 0.0, vz_left = 0.0;
    if (MAGNETIC_FIELDS_ENABLED) {
        bbx_left = pin->GetReal("problem", "bxl");
        bby_left = pin->GetReal("problem", "byl");
        bbz_left = pin->GetReal("problem", "bzl");
    }

    // right side of interface
    pgas_right = pgas_left;
    rho_right = rho_left*rho_ratio;
    bbx_right = 0.0, bby_right = 0.0, bbz_right = 0.0;
    vx_right = 0.0, vy_right = 0.0, vz_right = 0.0;

    // Jet parameters
    d_jet      = pin->GetReal("problem", "d_jet");
    p_jet      = pin->GetReal("problem", "p_jet");
    vx_jet     = pin->GetReal("problem", "vx_jet");
    vy_jet     = pin->GetReal("problem", "vy_jet");
    vz_jet     = pin->GetReal("problem", "vz_jet");
    if (MAGNETIC_FIELDS_ENABLED) {
        bx_jet = pin->GetReal("problem", "bx_jet");
        by_jet = pin->GetReal("problem", "by_jet");
        bz_jet = pin->GetReal("problem", "bz_jet");
    }
    x_jet      = pin->GetReal("problem", "x_jet");
    r_jet      = pin->GetReal("problem", "r_jet");
    x3_0 = 0.5*(mesh_size.x3max + mesh_size.x3min);

    // enroll boundary value function pointers
    EnrollUserBoundaryFunction(BoundaryFace::inner_x2, JetInnerX2);

    return;
}
//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   sets conserved variables according to input primitives
//   assigns fields based on cell-center positions, rather than interface positions

void MeshBlock::ProblemGenerator(ParameterInput *pin) {

  // Prepare auxiliary arrays
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ke+1, je+1, ie+1);

  // Initialize hydro variables
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        // Determine which variables to use
        Real rho = rho_left;
        Real pgas = pgas_left;
        Real vx = vx_left;
        Real vy = vy_left;
        Real vz = vz_left;
        Real bbx = bbx_left;
        Real bby = bby_left;
        Real bbz = bbz_left;
	    Real icm_pos = icm_pos0 + pcoord->x2v(j)*tan_icm;
        bool right_side = pcoord->x1v(i) > icm_pos;
        if (right_side) {
          rho = rho_right;
          pgas = pgas_right;
          vx = vx_right;
          vy = vy_right;
          vz = vz_right;
          bbx = bbx_right;
          bby = bby_right;
          bbz = bbz_right;
        }

        // Construct 4-vectors
        Real ut = std::sqrt(1.0 / (1.0 - (SQR(vx)+SQR(vy)+SQR(vz))));
        Real ux = ut * vx;
        Real uy = ut * vy;
        Real uz = ut * vz;
        Real bt = bbx*ux + bby*uy + bbz*uz;
        Real bx = (bbx + bt * ux) / ut;
        Real by = (bby + bt * uy) / ut;
        Real bz = (bbz + bt * uz) / ut;

        // Transform 4-vectors
        Real u0, u1, u2, u3;
        Real b0, b1, b2, b3;
        u0 = ut;
        u1 = ux;
        u2 = uy;
        u3 = uz;
        b0 = bt;
        b1 = bx;
        b2 = by;
        b3 = bz;

        // Set primitives
        phydro->w(IDN,k,j,i) = phydro->w1(IDN,k,j,i) = rho;
        phydro->w(IPR,k,j,i) = phydro->w1(IPR,k,j,i) = pgas;
	    phydro->w(IVX,k,j,i) = phydro->w1(IM1,k,j,i) = u1 / u0;
	    phydro->w(IVY,k,j,i) = phydro->w1(IM2,k,j,i) = u2 / u0;
	    phydro->w(IVZ,k,j,i) = phydro->w1(IM3,k,j,i) = u3 / u0;
        
        // Set magnetic fields
        bb(IB1,k,j,i) = b1 * u0 - b0 * u1;
        bb(IB2,k,j,i) = b2 * u0 - b0 * u2;
        bb(IB3,k,j,i) = b3 * u0 - b0 * u3;
      }
    }
  }
  peos->PrimitiveToConserved(phydro->w, bb, phydro->u, pcoord, is, ie, js, je, ks, ke);

  // Initialize magnetic field
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=js; j<=je+1; ++j) {
        for (int i=is; i<=ie+1; ++i) {
          // Determine which variables to use
          Real vx = vx_left;
          Real vy = vy_left;
          Real vz = vz_left;
          Real bbx = bbx_left;
          Real bby = bby_left;
          Real bbz = bbz_left;
          Real icm_pos = icm_pos0 + pcoord->x2v(j)*tan_icm;
	      bool right_side = pcoord->x1v(i) > icm_pos;
          if (right_side) {
            vx = vx_right;
            vy = vy_right;
            vz = vz_right;
            bbx = bbx_right;
            bby = bby_right;
            bbz = bbz_right;
          }

          // Construct 4-vectors
          Real ut = std::sqrt(1.0 / (1.0 - (SQR(vx)+SQR(vy)+SQR(vz))));
          Real ux = ut * vx;
          Real uy = ut * vy;
          Real uz = ut * vz;
          Real bt = bbx*ux + bby*uy + bbz*uz;
          Real bx = (bbx + bt * ux) / ut;
          Real by = (bby + bt * uy) / ut;
          Real bz = (bbz + bt * uz) / ut;

          // Set magnetic fields
          Real u0, u1, u2, u3;
          Real b0, b1, b2, b3;
          if (j != je+1 && k != ke+1) {
            pfield->b.x1f(k,j,i) = bbx;
          }
          if (i != ie+1 && k != ke+1) {
            pfield->b.x2f(k,j,i) = bby;
          }
          if (i != ie+1 && j != je+1) {
            pfield->b.x3f(k, j, i) = bbz;
          }
        }
      }
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void JetInnerX2()
//  \brief Sets boundary condition on left X2 boundary for Cygnus A jet problem

void JetInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim, FaceField &b,
                Real time, Real dt,
                int il, int iu, int jl, int ju, int kl, int ku, int ngh) {
    // set primitive variables in inlet ghost zones
    for (int k=kl; k<=ku; ++k) {
        for (int j=1; j<=ngh; ++j) {
            for (int i=il; i<=iu; ++i) {
                Real rad = std::sqrt(SQR(pco->x1v(i)-x_jet) + SQR(pco->x3v(k)-x3_0));
                if (rad <= r_jet) {
                    prim(IDN,k,jl-j,i) = d_jet;
                    prim(IVX,k,jl-j,i) = vx_jet;
                    prim(IVY,k,jl-j,i) = vy_jet;
                    prim(IVZ,k,jl-j,i) = vz_jet;
                    prim(IPR,k,jl-j,i) = p_jet;
                } else {
                    prim(IDN,k,jl-j,i) = prim(IDN,k,jl,i);
                    prim(IVX,k,jl-j,i) = prim(IVX,k,jl,i);
                    prim(IVY,k,jl-j,i) = prim(IVY,k,jl,i);
                    prim(IVZ,k,jl-j,i) = prim(IVZ,k,jl,i);
                    prim(IPR,k,jl-j,i) = prim(IPR,k,jl,i);
                }
            }
        }
    }

    // set magnetic field in inlet ghost zones
    if (MAGNETIC_FIELDS_ENABLED) {
        for (int k=kl; k<=ku; ++k) {
            for (int j=1; j<=ngh; ++j) {
#pragma omp simd
                for (int i=il; i<=iu+1; ++i) {
                    Real rad = std::sqrt(SQR(pco->x1v(i)-x_jet) + SQR(pco->x3v(k)-x3_0));
                    if (rad <= r_jet) {
                        b.x1f(k,jl-j,i) = bx_jet;
                    } else {
                        b.x1f(k,jl-j,i) = b.x1f(k,jl,i);
                    }
                }
            }
        }

        for (int k=kl; k<=ku; ++k) {
            for (int j=1; j<=ngh; ++j) {
#pragma omp simd
                for (int i=il; i<=iu; ++i) {
                    Real rad = std::sqrt(SQR(pco->x1v(i)-x_jet) + SQR(pco->x3v(k)-x3_0));
                    if (rad <= r_jet) {
                        b.x2f(k,jl-j,i) = by_jet;
                    } else {
                        b.x2f(k,jl-j,i) = b.x2f(k,jl,i);
                    }
                }
            }
        }

        for (int k=kl; k<=ku+1; ++k) {
            for (int j=1; j<=ngh; ++j) {
#pragma omp simd
                for (int i=il; i<=iu; ++i) {
                    Real rad = std::sqrt(SQR(pco->x1v(i)-x_jet) + SQR(pco->x3v(k)-x3_0));
                    if (rad <= r_jet) {
                        b.x3f(k,jl-j,i) = bz_jet;
                    } else {
                        b.x3f(k,jl-j,i) = b.x3f(k,jl,i);
                    }
                }
            }
        }
    }
}
