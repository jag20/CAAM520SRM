#include <petsc/private/snesimpl.h> /*I "petscsnes.h" I*/
#include <petscsnessrmg.h>

#include <petsc/private/dmimpl.h> /* need dm->setupcalled */
#include <petsc/private/dmdaimpl.h> /* need dm->Nlocal */
#include <petscdmda.h>

static PetscBool SRMGcite = PETSC_FALSE;
static const char SRMGCitation[] = \
  "@article{AdamsBrownKnepleySamtaney2016,\n"
  "  title   = {Segmental Refinement: A Multigrid Technique for Data Locality},\n"
  "  author  = {Mark F. Adams and Jed Brown and Matt Knepley and Ravi Samtaney},\n"
  "  journal = {SIAM Journal on Scientific Computing},\n"
  "  url     = {http://arxiv.org/abs/1406.7808},\n"
  "  note    = {to appear},\n"
  "  year    = {2016}\n}\n";

typedef struct {
  PetscBool setfromopts; /* Flag indicating that this object has been configured from options */
  SNES      solCoarse;   /* Solver for the coarse grid */
  SNES      solPatch;    /* Solver for patch grids */
  PetscInt  numLevels;   /* Number of refinement levels, 0 means solve only on coarse grid */
  PetscInt  r;           /* The refinement ratio */
} SNES_SRMG;

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGGetBounds_Static"
PetscErrorCode SNESSRMGGetBounds_Static(PetscInt quadrant, PetscInt buffer, PetscInt mask, PetscInt S, PetscInt s, PetscInt *pS, PetscInt *ps, PetscInt M, PetscInt m, PetscInt *pM, PetscInt *pm, PetscReal cs, PetscReal *pcs, PetscReal ce, PetscReal *pce)
{
  PetscFunctionBegin;
  if (quadrant & mask) {
    *ps  = s + (m-1)/2;
    *pS  = PetscMax(*ps - buffer, S);
    *pm  = m - (m-1)/2;
    *pM  = (*ps-*pS) + PetscMin(*pm + buffer, S+M - *ps);
    *pcs = cs + (M > 1 ? (*pS - S)*(ce-cs)/(M-1) : 0.0);
    *pce = ce;
  } else {
    *ps  = s;
    *pS  = PetscMax(s - buffer, S);
    *pm  = (m+1)/2;
    *pM  = (*ps-*pS) + PetscMin(*pm + buffer, S+M - *ps);
    *pcs = cs;
    *pce = cs + (m > 1 ? ((*pM)-1)*(ce-cs)/(M-1) : ce);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGCreatePatch_Static"
static PetscErrorCode SNESSRMGCreatePatch_Static(DM dm, PetscInt quadrant, PetscInt r, PetscInt buffer, Mat *interp, DM *patch)
{
  DMDAStencilType stencil_type;
  Mat             in;
  Vec             coords;
  PetscInt        debug = 0, dim, dof, s, i, j, k, *dnz;
  PetscInt        xs,  xm,  ys,  ym,  zs,  zm,  Xs,  Xm, Xo,  Ys,  Ym, Yo,  Zs,  Zm, Zo;
  PetscInt        pxs, pxm, pys, pym, pzs, pzm, pXs, pXm, pYs, pYm, pZs, pZm;
  PetscReal       cxs = 0.0, cxe = 1.0, cys = 0.0, cye = 1.0, czs = 0.0, cze = 1.0;
  PetscReal       pcxs, pcxe, pcys, pcye, pczs, pcze;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  if (buffer < 0) SETERRQ1(comm, PETSC_ERR_SUP, "Buffer size must be non-negative, not %d", buffer);
  ierr = PetscOptionsGetInt(NULL, NULL, "-snes_srmg_patch_debug", &debug, NULL);CHKERRQ(ierr);
  ierr = DMDACreate(PETSC_COMM_SELF, patch);CHKERRQ(ierr);
  ierr = DMAppendOptionsPrefix(*patch, "srmg_patch_");CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMSetDimension(*patch, dim);CHKERRQ(ierr);
  ierr = DMDAGetDof(dm, &dof);CHKERRQ(ierr);
  ierr = DMDASetDof(*patch, dof);CHKERRQ(ierr);
  ierr = DMDAGetStencilType(dm, &stencil_type);CHKERRQ(ierr);
  ierr = DMDASetStencilType(*patch, stencil_type);CHKERRQ(ierr);
  ierr = DMDAGetStencilWidth(dm, &s);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*patch, s);CHKERRQ(ierr);
  /* Determine patch */
  ierr = DMDAGetCorners(dm, &Xs, &Ys, &Zs, &Xm, &Ym, &Zm);CHKERRQ(ierr);
  ierr = DMDAGetOverlap(dm, &Xo, &Yo, &Zo);CHKERRQ(ierr);
  Xs += Xo; Ys += Yo; Zs += Zo;
  ierr = DMDAGetNonOverlappingRegion(dm, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  if (!xm || !ym || !zm) {xs = Xs; xm = Xm; ys = Ys; ym = Ym; zs = Zs; zm = Zm;}
  ierr = DMGetCoordinates(dm, &coords);CHKERRQ(ierr);
  if (dim > 0) {ierr = VecStrideMin(coords, 0, NULL, &cxs);CHKERRQ(ierr);ierr = VecStrideMax(coords, 0, NULL, &cxe);CHKERRQ(ierr);}
  if (dim > 1) {ierr = VecStrideMin(coords, 1, NULL, &cys);CHKERRQ(ierr);ierr = VecStrideMax(coords, 1, NULL, &cye);CHKERRQ(ierr);}
  if (dim > 2) {ierr = VecStrideMin(coords, 2, NULL, &czs);CHKERRQ(ierr);ierr = VecStrideMax(coords, 2, NULL, &cze);CHKERRQ(ierr);}
  ierr = SNESSRMGGetBounds_Static(quadrant, buffer, 0x1, Xs, xs, &pXs, &pxs, Xm, xm, &pXm, &pxm, cxs, &pcxs, cxe, &pcxe);CHKERRQ(ierr);
  ierr = SNESSRMGGetBounds_Static(quadrant, buffer, 0x2, Ys, ys, &pYs, &pys, Ym, ym, &pYm, &pym, cys, &pcys, cye, &pcye);CHKERRQ(ierr);
  ierr = SNESSRMGGetBounds_Static(quadrant, buffer, 0x4, Zs, zs, &pZs, &pzs, Zm, zm, &pZm, &pzm, czs, &pczs, cze, &pcze);CHKERRQ(ierr);
  if (debug) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "X: [%d, %d) [%d, %d) (%g, %g)\n", pxs, pxs+pxm, pXs, pXs+pXm, pcxs, pcxe);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Y: [%d, %d) [%d, %d) (%g, %g)\n", pys, pys+pym, pYs, pYs+pYm, pcys, pcye);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF, "Z: [%d, %d) [%d, %d) (%g, %g)\n", pzs, pzs+pzm, pZs, pZs+pZm, pczs, pcze);CHKERRQ(ierr);
  }
  /* Refine grid */
  pXm  = r*(pXm - 1) + 1; pxm = r*(pxm - 1) + 1;
  pYm  = r*(pYm - 1) + 1; pym = r*(pym - 1) + 1;
  pZm  = r*(pZm - 1) + 1; pzm = r*(pzm - 1) + 1;
  pXs  = r*pXs; pYs = r*pYs; pZs = r*pZs;
  pxs  = r*pxs; pys = r*pys; pzs = r*pzs;
  ierr = DMDASetSizes(*patch, pXm, pYm, pZm);CHKERRQ(ierr);
  ierr = DMDASetNonOverlappingRegion(*patch, pxs, pys, pzs, pxm, pym, pzm);CHKERRQ(ierr);
  ierr = DMDASetOverlap(*patch, pXs, pYs, pZs);CHKERRQ(ierr);
  ierr = DMCopyDMSNES(dm, *patch);CHKERRQ(ierr);
  ierr = DMSetUp(*patch);CHKERRQ(ierr);
  /*   Create interpolator from coarse global vector to patch local (interior+buffer) vector */
  ierr = PetscCalloc1(Xm*Ym*Zm, &dnz);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &in);CHKERRQ(ierr);
  ierr = MatSetSizes(in, Xm*Ym*Zm, pXm*pYm*pZm, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(in, MATAIJ);CHKERRQ(ierr);
  for (k = pZs; k < pZs+pZm; ++k) {
    for (j = pYs; j < pYs+pYm; ++j) {
      for (i = pXs; i < pXs+pXm; ++i) {
        const PetscInt go = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r -  Xs);
        PetscInt       g[4];

        if (i % r) {
          if (j % r) {
            g[0] = go;
            g[1] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r+1 -  Xs);
            g[2] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r -  Xs);
            g[3] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r+1 -  Xs);
            dnz[g[0]]++; dnz[g[1]]++; dnz[g[2]]++; dnz[g[3]]++;
          } else {
            g[0] = go;  g[1] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r+1 -  Xs);
            dnz[g[0]]++; dnz[g[1]]++;
          }
        } else {
          if (j % r) {
            g[0] = go;  g[1] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r -  Xs);
            dnz[g[0]]++; dnz[g[1]]++;
          } else {
            dnz[go]++;
          }
        }
      }
    }
  }
  ierr = MatXAIJSetPreallocation(in, 1, dnz, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = MatSetUp(in);CHKERRQ(ierr);
  for (k = pZs; k < pZs+pZm; ++k) {
    for (j = pYs; j < pYs+pYm; ++j) {
      for (i = pXs; i < pXs+pXm; ++i) {
        const PetscInt lo = ((k - pZs)*pYm + (j - pYs))*pXm + (i - pXs);
        const PetscInt go = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r -  Xs);
        PetscInt       g[4];
        PetscScalar    v[4];

        if (i % r) {
          if (j % r) {
            g[0] = go;
            g[1] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r+1 -  Xs);
            g[2] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r -  Xs);
            g[3] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r+1 -  Xs);
            v[0] = 0.25; v[1] = 0.25; v[2] = 0.25; v[3] = 0.25;
            ierr = MatSetValues(in, 4, g, 1, &lo, v, INSERT_VALUES);CHKERRQ(ierr);
          } else {
            g[0] = go;  g[1] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r+1 -  Xs);
            v[0] = 0.5; v[1] = 0.5;
            ierr = MatSetValues(in, 2, g, 1, &lo, v, INSERT_VALUES);CHKERRQ(ierr);
          }
        } else {
          if (j % r) {
            g[0] = go;  g[1] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r -  Xs);
            v[0] = 0.5; v[1] = 0.5;
            ierr = MatSetValues(in, 2, g, 1, &lo, v, INSERT_VALUES);CHKERRQ(ierr);
          } else {
            v[0] = 1.0;
            ierr = MatSetValues(in, 1, &go, 1, &lo, v, INSERT_VALUES);CHKERRQ(ierr);
          }
        }
      }
    }
  }
  ierr = MatAssemblyBegin(in, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(in, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCreateMAIJ(in, dof, interp);CHKERRQ(ierr);
  ierr = MatDestroy(&in);CHKERRQ(ierr);
  ierr = PetscFree(dnz);CHKERRQ(ierr);
  /* Scatter in coordinates */
  ierr = DMDASetUniformCoordinates(*patch, pcxs, pcxe, pcys, pcye, pczs, pcze);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*patch, NULL, "-dm_view");CHKERRQ(ierr);
  dm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_SRMG"
static PetscErrorCode SNESSetUp_SRMG(SNES snes)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!snes->setupcalled) {
    DM  dm;
    KSP ksp;
    PC  pc;

    /* Create patch solver */
    ierr = SNESCreate(PetscObjectComm((PetscObject) snes), &sr->solPatch);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject) sr->solPatch, (PetscObject) snes, 1);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(sr->solPatch, "srmg_patch_");CHKERRQ(ierr);
    /* Create coarse solver */
    ierr = SNESCreate(PetscObjectComm((PetscObject) snes), &sr->solCoarse);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject) sr->solCoarse, (PetscObject) snes, 1);CHKERRQ(ierr);
    ierr = SNESAppendOptionsPrefix(sr->solCoarse, "srmg_coarse_");CHKERRQ(ierr);
    ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
    ierr = SNESSetDM(sr->solCoarse, dm);CHKERRQ(ierr);
    /*   coarse solve is (redundant) LU by default; set shifttype NONZERO to avoid annoying zero-pivot in LU preconditioner */
    ierr = SNESSetType(sr->solCoarse, SNESNEWTONLS);CHKERRQ(ierr);
    ierr = SNESGetKSP(sr->solCoarse, &ksp);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject) ksp, (PetscObject) sr->solCoarse, 1);CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject) snes), &size);CHKERRQ(ierr);
    if (size > 1) {
      KSP innerksp;
      PC  innerpc;

      ierr = PCSetType(pc, PCREDUNDANT);CHKERRQ(ierr);
      ierr = PCRedundantGetKSP(pc, &innerksp);CHKERRQ(ierr);
      ierr = KSPGetPC(innerksp, &innerpc);CHKERRQ(ierr);
      ierr = PCFactorSetShiftType(innerpc, MAT_SHIFT_INBLOCKS);CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pc, PCLU);CHKERRQ(ierr);
      ierr = PCFactorSetShiftType(pc, MAT_SHIFT_INBLOCKS);CHKERRQ(ierr);
    }
  }
  if (snes->pcside == PC_LEFT && snes->functype == SNES_FUNCTION_DEFAULT) snes->functype = SNES_FUNCTION_PRECONDITIONED;
  if (sr->setfromopts) {ierr = SNESSetFromOptions(sr->solCoarse);CHKERRQ(ierr);}
  ierr = SNESSetUp(sr->solCoarse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGInjectSolution_Static"
PetscErrorCode SNESSRMGInjectSolution_Static(DM dmPatch, Vec uPatch, PetscInt r, DM dmCoarse, Vec solCoarse)
{
  PetscScalar   *px, *x;
  PetscInt       dof, pXs, pXm, pYs, pYm, pZs, pZm, Xs, Xm, Ys, Ym, Zs, Zm, Xo, Yo, Zo;
  PetscInt       i, j, k, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDAGetDof(dmPatch, &dof);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dmPatch, &pXs, &pYs, &pZs, &pXm, &pYm, &pZm);CHKERRQ(ierr);
  ierr = DMDAGetOverlap(dmPatch, &Xo, &Yo, &Zo);CHKERRQ(ierr);
  pXs += Xo; pYs += Yo; pZs += Zo;
  ierr = DMDAGetCorners(dmCoarse, &Xs, &Ys, &Zs, &Xm, &Ym, &Zm);CHKERRQ(ierr);
  ierr = DMDAGetOverlap(dmCoarse, &Xo, &Yo, &Zo);CHKERRQ(ierr);
  Xs += Xo; Ys += Yo; Zs += Zo;
  ierr = VecGetArray(uPatch, &px);CHKERRQ(ierr);
  ierr = VecGetArray(solCoarse, &x);CHKERRQ(ierr);
  for (k = pZs; k < pZs+pZm; ++k) {
    for (j = pYs; j < pYs+pYm; ++j) {
      for (i = pXs; i < pXs+pXm; ++i) {
        const PetscInt lo = ((k - pZs)*pYm + (j - pYs))*pXm + (i - pXs);
        const PetscInt go = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r -  Xs);

        if (!(k%r) && !(j%r) && !(i%r)) for (d = 0; d < dof; ++d) x[go+d] = px[lo+d];
      }
    }
  }
  ierr = VecRestoreArray(uPatch, &px);CHKERRQ(ierr);
  ierr = VecRestoreArray(solCoarse, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGProcessLevel_Static"
PetscErrorCode SNESSRMGProcessLevel_Static(SNES_SRMG *sr, DM dmCoarse, Vec solCoarse, PetscInt remainingLevels, PetscInt buffer)
{
  PetscInt       numQuadrants = 1, q, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!remainingLevels) PetscFunctionReturn(0);
  ierr = DMGetDimension(dmCoarse, &dim);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) numQuadrants *= 2;
  for (q = 0; q < numQuadrants; ++q) {
    DM  dmPatch;
    Mat interp;
    Vec uPatch, bcPatch;

    ierr = SNESReset(sr->solPatch);CHKERRQ(ierr);
    ierr = SNESSRMGCreatePatch_Static(dmCoarse, q, sr->r, buffer, &interp, &dmPatch);CHKERRQ(ierr);
    ierr = SNESSetDM(sr->solPatch, dmPatch);CHKERRQ(ierr);
    if (sr->setfromopts) {ierr = SNESSetFromOptions(sr->solPatch);CHKERRQ(ierr);}
    ierr = DMGetGlobalVector(dmPatch, &uPatch);CHKERRQ(ierr);
    /* TODO Must interpolate rhs and solution vector */
    ierr = MatMultTranspose(interp, solCoarse, uPatch);CHKERRQ(ierr);
    ierr = VecViewFromOptions(uPatch, (PetscObject) sr->solPatch, "-input_view");CHKERRQ(ierr);
    /* Setup patch boundary conditions */
    ierr = DMGetNamedLocalVector(dmPatch, "_petsc_boundary_conditions_", &bcPatch);CHKERRQ(ierr);
    ierr = VecCopy(uPatch, bcPatch);CHKERRQ(ierr);
    ierr = DMRestoreNamedLocalVector(dmPatch, "_petsc_boundary_conditions_", &bcPatch);CHKERRQ(ierr);
    ierr = SNESSolve(sr->solPatch, NULL, uPatch);CHKERRQ(ierr);
    /* Recurse onto finer level */
    /* TODO Determine buffer for fine level */
    ierr = SNESSRMGProcessLevel_Static(sr, dmPatch, uPatch, remainingLevels-1, buffer);CHKERRQ(ierr);
    /* TODO Process patch solution: We can use the SNES monitor setup to manage functionals */
    ierr = SNESSRMGInjectSolution_Static(dmPatch, uPatch, sr->r, dmCoarse, solCoarse);CHKERRQ(ierr);
    /* TODO Pass up patch solver convergence */
    /* Cleanup */
    ierr = DMRestoreGlobalVector(dmPatch, &uPatch);CHKERRQ(ierr);
    ierr = MatDestroy(&interp);CHKERRQ(ierr);
    ierr = DMDestroy(&dmPatch);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_SRMG"
static PetscErrorCode SNESSolve_SRMG(SNES snes)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  DM             dmCoarse;
  PetscInt       buffer = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject) snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject) snes)->type_name);
  ierr = PetscCitationsRegister(SRMGCitation, &SRMGcite);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-buffer_size", &buffer, NULL);CHKERRQ(ierr);

  ierr = SNESSolve(sr->solCoarse, snes->vec_rhs, snes->vec_sol);CHKERRQ(ierr);
  ierr = SNESGetDM(sr->solCoarse, &dmCoarse);CHKERRQ(ierr);
  ierr = SNESSRMGProcessLevel_Static(sr, dmCoarse, snes->vec_sol, sr->numLevels, buffer);CHKERRQ(ierr);
  {
    SNESConvergedReason reason;

    ierr = SNESGetConvergedReason(sr->solCoarse, &reason);CHKERRQ(ierr);
    ierr = SNESSetConvergedReason(snes, reason);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_SRMG"
static PetscErrorCode SNESReset_SRMG(SNES snes)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sr->setfromopts = PETSC_FALSE;
  ierr = SNESDestroy(&sr->solCoarse);CHKERRQ(ierr);
  ierr = SNESDestroy(&sr->solPatch);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_SRMG"
static PetscErrorCode SNESDestroy_SRMG(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESReset_SRMG(snes);CHKERRQ(ierr);
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_SRMG"
static PetscErrorCode SNESSetFromOptions_SRMG(PetscOptionItems *PetscOptionsObject, SNES snes)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sr->setfromopts = PETSC_TRUE;
  ierr = PetscOptionsHead(PetscOptionsObject, "SRMG options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_srmg_levels", "How many refinements to make", "SNESSRMGSetNumLevels", sr->numLevels, &sr->numLevels, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_srmg_refinement_ratio", "How much to refine patch", "SNESSRMGSetRefinementRatio", sr->r, &sr->r, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_SRMG"
PetscErrorCode SNESView_SRMG(SNES snes, PetscViewer viewer)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  SRMG with %D levels\n", sr->numLevels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "SRMG Coarse grid solver ----------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = SNESView(sr->solCoarse, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    if (sr->numLevels) {
      ierr = PetscViewerASCIIPrintf(viewer, "SRMG Patch grid solver ----------------------------------------\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = SNESView(sr->solPatch, viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*MC
  SNESSRMG - Segmental Refinement Multigrid (i.e. low-memory solver)

  Options Database Key:
. -snes_srmg_levels <nlevels> - number of refinement levels

  Level: beginner

  Concepts: segmental refinement, multigrid, preconditioners

.seealso: SNESCreate(), SNESSetType(), SNESType (for list of available types), SNES
M*/
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_SRMG"
PETSC_EXTERN PetscErrorCode SNESCreate_SRMG(SNES snes)
{
  SNES_SRMG     *sr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(snes, &sr);CHKERRQ(ierr);
  snes->data = (void *) sr;

  sr->setfromopts = PETSC_FALSE;
  sr->solCoarse   = NULL;
  sr->numLevels   = 0;
  sr->r           = 1;

  snes->ops->setup          = SNESSetUp_SRMG;
  snes->ops->solve          = SNESSolve_SRMG;
  snes->ops->destroy        = SNESDestroy_SRMG;
  snes->ops->setfromoptions = SNESSetFromOptions_SRMG;
  snes->ops->view           = SNESView_SRMG;
  snes->ops->reset          = SNESReset_SRMG;

  snes->pcside  = PC_RIGHT;
  snes->usesksp = PETSC_TRUE;
  snes->usespc  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SRMGInitializePackage"
PetscErrorCode SRMGInitializePackage()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESRegister(SNESSRMG, SNESCreate_SRMG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGSetNumLevels"
/*@
  SNESSRMGSetNumLevels - Sets the number of refinement levels

  Logically Collective on SNES

  Input Parameters:
+ snes - the solver context
- n    - the number of levels

  Options Database Key:
. -snes_srmg_levels <nlevels>

  Level: intermediate

  Concepts: SRMG preconditioner

.seealso: SNESSRMGGetNumLevels()
@*/
PetscErrorCode SNESSRMGSetNumLevels(SNES snes, PetscInt n)
{
  SNES_SRMG *sr = (SNES_SRMG *) snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  sr->numLevels = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGGetNumLevels"
/*@
  SNESSRMGGetNumLevels - Gets the number of refinement levels

  Not Collective on SNES

  Input Parameter:
. snes - the solver context

  Output Parameter:
. n - the number of levels

  Level: intermediate

  Concepts: SRMG preconditioner

.seealso: SNESSRMGSetNumLevels()
@*/
PetscErrorCode SNESSRMGGetNumLevels(SNES snes, PetscInt *n)
{
  SNES_SRMG *sr = (SNES_SRMG *) snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidPointer(n, 2);
  *n = sr->numLevels;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGSetRefinementRatio"
/*@
  SNESSRMGSetRefinementRatio - Sets the refinement ratio

  Logically Collective on SNES

  Input Parameters:
+ snes - the solver context
- r    - the refinement ratio, e.g. 2 for doubling of the grid in each dimension

  Options Database Key:
. -snes_srmg_refinement_ratio <r>

  Level: intermediate

  Concepts: SRMG preconditioner

.seealso: SNESSRMGGetRefinementRatio()
@*/
PetscErrorCode SNESSRMGSetRefinementRatio(SNES snes, PetscInt r)
{
  SNES_SRMG *sr = (SNES_SRMG *) snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  sr->r = r;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGGetRefinementRatio"
/*@
  SNESSRMGGetRefinementRatio - Gets the refinement ratio

  Not Collective on SNES

  Input Parameter:
. snes - the solver context

  Output Parameter:
. r - the refinement ratio, e.g. 2 for doubling of the grid in each dimension

  Level: intermediate

  Concepts: SRMG preconditioner

.seealso: SNESSRMGSetRefinementRatio()
@*/
PetscErrorCode SNESSRMGGetRefinementRatio(SNES snes, PetscInt *r)
{
  SNES_SRMG *sr = (SNES_SRMG *) snes->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidPointer(r, 2);
  *r = sr->r;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_srmg"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This registers all of the SNES methods that are in the basic PETSc libpetscsnes library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_srmg(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SRMGInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
