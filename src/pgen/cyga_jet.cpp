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

//----------------------------------------------------------------------------------------
// Function for setting initial conditions
// Inputs:
//   pin: parameters
// Outputs: (none)
// Notes:
//   sets conserved variables according to input primitives
//   assigns fields based on cell-center positions, rather than interface positions

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  // Read and set ratio of specific heats
  Real gamma_adi = peos->GetGamma();
  Real gamma_adi_red = gamma_adi / (gamma_adi - 1.0);

  // Read and check interface position position
  //int shock_dir = pin->GetInteger("problem", "shock_dir");
  Real icm_pos0 = pin->GetReal("problem", "icm_pos0");
  Real icm_angle = pin->GetReal("problem", "icm_angle");
  icm_angle *= PI/180;
  Real tan_icm = std::tan(icm_angle);

  std::stringstream msg;

  if (icm_pos0 < pmy_mesh->mesh_size.x1min || icm_pos0 > pmy_mesh->mesh_size.x1max) {
    msg << "### FATAL ERROR in Problem Generator\n"
        << "icm_pos0=" << icm_pos0 << " lies outside domain" << std::endl;
    ATHENA_ERROR(msg);
  }

  // Read parameters
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

  // Prepare auxiliary arrays
  AthenaArray<Real> bb;
  bb.NewAthenaArray(3, ke+1, je+1, ie+1);

  // Set right state
  Real pgas_right = pgas_left;
  Real rho_right = rho_left*rho_ratio;
  Real bbx_right = 0.0, bby_right = 0.0, bbz_right = 0.0;
  Real vx_right = 0.0, vy_right = 0.0, vz_right = 0.0;

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
        {
          u0 = ut;
          u1 = ux;
          u2 = uy;
          u3 = uz;
          b0 = bt;
          b1 = bx;
          b2 = by;
          b3 = bz;
        }

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
          {
            if (j != je+1 && k != ke+1) {
              pfield->b.x1f(k,j,i) = bbx;
            }
            if (i != ie+1 && k != ke+1) {
              pfield->b.x2f(k,j,i) = bby;
            }
            if (i != ie+1 && j != je+1) {
              pfield->b.x3f(k,j,i) = bbz;
            }
          }
        }
      }
    }
  }
  return;
}
