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
  PetscBool inject;      /* Flag to inject fine solution to the coarse problem after solve */
  PetscInt  interpOrder; /* The interpolation order */
} SNES_SRMG;

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGGetBounds_Static"
/*
  SNESSRMGGetBounds_Static - We divide a dimension in half and return a patch covering that half with the specified buffer. The mask tells us whether to return the left or right half.

  Not collective

  Input Parameters:
+ quadrant - The quadrant number, in 3D its [0, 8)
. buffer   - The buffer size around the patch
. S        - The start of the parent
. s        - The start of the interior of the parent, if S == s then there is no buffer around the parent on the left side
. M        - The size of the parent
. m        - The size of the parent interior, if S+M == s+m then there is no buffer around the parent on the right side
. cs       - The coordinate of the left side of the parent (vertex S)
- ce       - The coordinate of the right side of the parent (vertex S+M)

  Output Parameters
+ pS       - The start of the patch
. ps       - The start of the patch interior, if pS == ps then there is no buffer around the patch on the left side
. pM       - The size of the patch
. pm       - The size of the patch interior, if pS+pM == ps+pm then there is no buffer around the patch on the right side
. pcs      - The coordinate of the left side of the patch (vertex pS)
- pce      - The coordinate of the right side of the patch (vertex pS+pM)

  Level: developer

.seealso: SNESSRMGCreatePatch_Static()
*/
PetscErrorCode SNESSRMGGetBounds_Static(PetscInt quadrant, PetscInt buffer, PetscInt mask, PetscInt S, PetscInt s, PetscInt *pS, PetscInt *ps, PetscInt M, PetscInt m, PetscInt *pM, PetscInt *pm, PetscReal cs, PetscReal *pcs, PetscReal ce, PetscReal *pce)
{
  const PetscReal h = M <= 1 ? 1.0 : (ce-cs)/(M-1);

  PetscFunctionBegin;
  if (quadrant & mask) {
    *ps  = s + (m-1)/2;
    *pS  = PetscMax(*ps - buffer, S);
    *pm  = m - (m-1)/2;
    *pM  = (*ps-*pS) + PetscMin(*pm + buffer, S+M - *ps);
    /* The coordinates are for the buffered parent region, and the buffer may be different from the child buffer */
    *pcs = cs + (M > 1 ? (*pS - S)*h : 0.0);
    *pce = ce - PetscMax(0, (S+M) - (s+m) - buffer) * h;
  } else {
    *ps  = s;
    *pS  = PetscMax(s - buffer, S);
    *pm  = (m+1)/2;
    *pM  = (*ps-*pS) + PetscMin(*pm + buffer, S+M - *ps);
    /* The coordinates are for the buffered parent region, and the buffer may be different from the child buffer */
    *pcs = cs + PetscMax(0, s - S - buffer) * h;
    *pce = (m > 1 ? *pcs + ((*pM)-1)*h : ce);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGCreateInterpolator_Linear_Static"
/* Create interpolator from coarse global vector to patch local (interior+buffer) vector */
static PetscErrorCode SNESSRMGCreateInterpolator_Linear_Static(PetscInt Xs, PetscInt Ys, PetscInt Zs, PetscInt Xm, PetscInt Ym, PetscInt Zm, PetscInt pXs, PetscInt pYs, PetscInt pZs, PetscInt pXm, PetscInt pYm, PetscInt pZm, PetscInt r, PetscInt dof, Mat *interp)
{
  Mat            in;
  PetscInt      *dnz;
  PetscInt       i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGCreateInterpolator_Quadratic_Static"
/* Create interpolator from coarse global vector to patch local (interior+buffer) vector */
static PetscErrorCode SNESSRMGCreateInterpolator_Quadratic_Static(PetscInt Xs, PetscInt Ys, PetscInt Zs, PetscInt Xm, PetscInt Ym, PetscInt Zm, PetscInt pXs, PetscInt pYs, PetscInt pZs, PetscInt pXm, PetscInt pYm, PetscInt pZm, PetscInt r, PetscInt dof, Mat *interp)
{
  Mat            in;
  PetscInt      *dnz;
  PetscInt       i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(Xm*Ym*Zm, &dnz);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &in);CHKERRQ(ierr);
  ierr = MatSetSizes(in, Xm*Ym*Zm, pXm*pYm*pZm, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(in, MATAIJ);CHKERRQ(ierr);
  for (k = pZs; k < pZs+pZm; ++k) {
    for (j = pYs; j < pYs+pYm; ++j) {
      for (i = pXs; i < pXs+pXm; ++i) {
        /*index of the representative coarse node (bottom left) in the patch*/
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
            if (pXm > 2){
              g[2] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r+2 -  Xs);
              if (i == pXs + pXm - 2) {
                g[2] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r-1 -  Xs);
                g[1] = g[0];
                g[0] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r+1 -  Xs);
              }
              dnz[g[2]]++;
            }
          }
        } else {
          if (j % r) {
            g[0] = go;  g[1] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r -  Xs);
            dnz[g[0]]++; dnz[g[1]]++;
            if (pYm > 2){
              g[2] = ((k/r -  Zs)*Ym  + (j/r+2 -  Ys))*Xm  + (i/r -  Xs);
              if (j == pYs + pYm - 2) {
                g[2] = ((k/r -  Zs)*Ym  + (j/r-1 -  Ys))*Xm  + (i/r -  Xs);
                g[1] = g[0];
                g[0] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r -  Xs);
              }
              dnz[g[2]]++;
            }
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
            if (pXm > 2){
              g[2] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r+2 -  Xs);
              if (i == pXs + pXm - 2) {
                g[2] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r-1 -  Xs);
                g[1] = g[0];
                g[0] = ((k/r -  Zs)*Ym  + (j/r -  Ys))*Xm  + (i/r+1 -  Xs);
              }
              v[0] = .375; v[1] = 0.75; v[2] = -.125;
              ierr = MatSetValues(in, 3, g, 1, &lo, v, INSERT_VALUES);CHKERRQ(ierr);
            } else{
              ierr = MatSetValues(in, 2, g, 1, &lo, v, INSERT_VALUES);CHKERRQ(ierr);
            }
          }
        } else {
          if (j % r) {
            g[0] = go;  g[1] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r -  Xs);
            v[0] = 0.5; v[1] = 0.5;
            if (pYm > 2){
              g[2] = ((k/r -  Zs)*Ym  + (j/r+2 -  Ys))*Xm  + (i/r -  Xs);
              if (j == pYs + pYm - 2) {
                g[2] = ((k/r -  Zs)*Ym  + (j/r-1 -  Ys))*Xm  + (i/r -  Xs);
                g[1] = g[0];
                g[0] = ((k/r -  Zs)*Ym  + (j/r+1 -  Ys))*Xm  + (i/r -  Xs);
              }
              v[0] = .375; v[1] = 0.75; v[2] = -.125;
              ierr = MatSetValues(in, 3, g, 1, &lo, v, INSERT_VALUES);CHKERRQ(ierr);
            } else {
              ierr = MatSetValues(in, 2, g, 1, &lo, v, INSERT_VALUES);CHKERRQ(ierr);
            }
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGCreatePatch_Static"
static PetscErrorCode SNESSRMGCreatePatch_Static(DM dm, PetscInt quadrant, PetscInt r, PetscInt interpOrder, PetscInt buffer, Mat *interp, DM *patch)
{
  DMDAStencilType stencil_type;
  Vec             coords;
  PetscInt        debug = 0, dim, dof, s;
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
  /* Create interpolator from coarse global vector to patch local (interior+buffer) vector */
  switch (interpOrder) {
  case 2:
    ierr = SNESSRMGCreateInterpolator_Quadratic_Static(Xs, Ys, Zs, Xm, Ym, Zm, pXs, pYs, pZs, pXm, pYm, pZm, r, dof, interp);CHKERRQ(ierr);break;
  case 1:
    ierr = SNESSRMGCreateInterpolator_Linear_Static(Xs, Ys, Zs, Xm, Ym, Zm, pXs, pYs, pZs, pXm, pYm, pZm, r, dof, interp);CHKERRQ(ierr);break;
  default:
    SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Interpolation order %D not supported", interpOrder);
  }
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

typedef struct {
  PetscInt  n, bn;
  PetscReal l1b,  l1;  /* Error in the $\ell_1$ norm on the full patch and the interior */
  PetscReal l2b,  l2;  /* Error in the $\ell_2$ norm on the full patch and the interior */
  PetscReal infb, inf; /* Error in the $\ell_\infty$ norm on the full patch and the interior */
  PetscErrorCode (*exact)(const PetscReal[], PetscScalar *, void *);
} ErrorCtx;

#undef __FUNCT__
#define __FUNCT__ "ExactSolution1_Static"
static PetscErrorCode ExactSolution1_Static(const PetscReal x[], PetscScalar *u, void *ctx)
{
  PetscFunctionBegin;
  *u = x[0]*(1.0 - x[0])*x[1]*(1.0 - x[1]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExactSolution2_Static"
static PetscErrorCode ExactSolution2_Static(const PetscReal x[], PetscScalar *u, void *ctx)
{
  PetscFunctionBegin;
  *u = PetscSinReal(PETSC_PI*x[0])*PetscSinReal(PETSC_PI*x[1]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGFunctionalCompute_Error"
PetscErrorCode SNESSRMGFunctionalCompute_Error(SNES snes, DM dmPatch, Vec uPatch, DM dmCoarse, Vec solCoarse, void *ctx)
{
  ErrorCtx      *e = (ErrorCtx *) ctx;
  DM             cdm;
  Vec            coordinates, error;
  DMDACoor2d   **coords;
  PetscScalar  **u, ue, **ev;
  PetscReal      x[3];
  PetscInt       xs, xm, xo, ys, ym, yo;
  PetscInt       xsi, xmi, ysi, ymi;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(dmPatch, &error);CHKERRQ(ierr);

  ierr = DMDAGetCorners(dmPatch, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
  ierr = DMDAGetOverlap(dmPatch, &xo, &yo, NULL);CHKERRQ(ierr);
  ierr = DMDAGetNonOverlappingRegion(dmPatch, &xsi, &ysi, NULL, &xmi, &ymi, NULL);CHKERRQ(ierr);
  xsi -= xo; ysi -= yo;
  ierr = DMGetCoordinateDM(dmPatch, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinates(dmPatch, &coordinates);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cdm, coordinates, &coords);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dmPatch, uPatch, &u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dmPatch, error, &ev);CHKERRQ(ierr);
  for (j = ys; j < ys+ym; ++j) {
    for (i = xs; i < xs+xm; ++i) {
      PetscReal diff;

      x[0] = PetscRealPart(coords[j][i].x); x[1] = PetscRealPart(coords[j][i].y);
      ierr = (*e->exact)(x, &ue, ctx);CHKERRQ(ierr);
      diff = PetscAbsScalar(u[j][i] - ue);
      e->l1b  += diff;
      e->l2b  += diff*diff;
      e->infb  = PetscMax(e->infb, diff);
      ++e->bn;
      ev[j][i] = diff;
      if (i >= xsi && i < xsi+xmi && j >= ysi && j < ysi+ymi) {
        /* Only include left and bottom boundaries of the patch at the boundary of the mesh */
        if ((i > xsi || !xo) && (j > ysi || !yo)) {
          ++e->n;
          e->l1  += diff;
          e->l2  += diff*diff;
          e->inf  = PetscMax(e->inf, diff);
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(dmPatch, uPatch, &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cdm, coordinates, &coords);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dmPatch, error, &ev);CHKERRQ(ierr);
  ierr = VecViewFromOptions(error, (PetscObject) dmPatch, "-error_view");CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmPatch, &error);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGFunctionalView_Error"
PetscErrorCode SNESSRMGFunctionalView_Error(SNES snes, PetscInt level, void *ctx)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  ErrorCtx      *e  = (ErrorCtx *) ctx;
  DM             dm;
  PetscInt       M, N, P, Nt, f = PetscPowInt(sr->r, level);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(sr->solCoarse, &dm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm, 0, &M, &N, &P, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  M    = f*(M-1) + 1;
  N    = f*(N-1) + 1;
  P    = f*(P-1) + 1;
  Nt   = M*N*P;
  ierr = PetscPrintf(PetscObjectComm((PetscObject) snes), "L: %D N: %D n: %D error l2 %g inf %g\n", level, Nt, e->n, (double) PetscSqrtReal(e->l2)/Nt, (double) e->inf);CHKERRQ(ierr);
  ierr = PetscPrintf(PetscObjectComm((PetscObject) snes), "L: %D Buffered error n: %D l2 %g inf %g\n", level, e->bn, (double) PetscSqrtReal(e->l2b)/Nt, (double) e->infb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSRMGInjectSolution_Static"
PetscErrorCode SNESSRMGInjectSolution_Static(SNES snes, DM dmPatch, Vec uPatch, DM dmCoarse, Vec solCoarse, void *ctx)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  PetscScalar   *px, *x;
  PetscInt       r = sr->r, dof, pXs, pXm, pYs, pYm, pZs, pZm, Xs, Xm, Ys, Ym, Zs, Zm, Xo, Yo, Zo;
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
PetscErrorCode SNESSRMGProcessLevel_Static(SNES snes, DM dmCoarse, Vec solCoarse, PetscInt remainingLevels, PetscInt buffer)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  ErrorCtx       ectx;
  PetscInt       numQuadrants = 1, q, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!remainingLevels) PetscFunctionReturn(0);
  /* TODO This should be passed in by the user when the functional is constructed, I think functionals should indicate the level on which they work and have a view function */
  if (1) {
    PetscInt mms;

    ectx.n = ectx.bn = 0;
    ectx.l1 = ectx.l1b = ectx.l2 = ectx.l2b = ectx.inf = ectx.infb = 0.0;
    ierr = PetscOptionsGetInt(NULL, NULL, "-mms", &mms, NULL);CHKERRQ(ierr);
    if (mms == 1) ectx.exact = ExactSolution1_Static;
    if (mms == 2) ectx.exact = ExactSolution2_Static;
  }
  ierr = DMGetDimension(dmCoarse, &dim);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) numQuadrants *= 2;
  ierr = PetscPrintf(PETSC_COMM_SELF, "L: %D Buffer %D\n", sr->numLevels - remainingLevels + 1, buffer);CHKERRQ(ierr);
  for (q = 0; q < numQuadrants; ++q) {
    DM  dmPatch;
    Mat interp;
    Vec uPatch, bcPatch;

    ierr = SNESReset(sr->solPatch);CHKERRQ(ierr);
    ierr = SNESSRMGCreatePatch_Static(dmCoarse, q, sr->r, sr->interpOrder, buffer, &interp, &dmPatch);CHKERRQ(ierr);
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
    ierr = SNESSRMGProcessLevel_Static(snes, dmPatch, uPatch, remainingLevels-1, buffer);CHKERRQ(ierr);
    /* TODO Process patch solution: We can use the SNES monitor setup to manage functionals */
    ierr = SNESSRMGFunctionalCompute_Error(snes, dmPatch, uPatch, dmCoarse, solCoarse, &ectx);CHKERRQ(ierr);
    if (sr->inject) {ierr = SNESSRMGInjectSolution_Static(snes, dmPatch, uPatch, dmCoarse, solCoarse, NULL);CHKERRQ(ierr);}
    /* TODO Pass up patch solver convergence */
    /* Cleanup */
    ierr = DMRestoreGlobalVector(dmPatch, &uPatch);CHKERRQ(ierr);
    ierr = MatDestroy(&interp);CHKERRQ(ierr);
    ierr = DMDestroy(&dmPatch);CHKERRQ(ierr);
  }
  ierr = SNESSRMGFunctionalView_Error(snes, sr->numLevels - remainingLevels + 1, &ectx);CHKERRQ(ierr);
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
  ierr = SNESSRMGProcessLevel_Static(snes, dmCoarse, snes->vec_sol, sr->numLevels, buffer);CHKERRQ(ierr);
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
  ierr = PetscOptionsBool("-snes_srmg_inject", "Inject fine solution into the coarse after patch solve", "SNESSRMGSetInject", sr->inject, &sr->inject, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-snes_srmg_interp_order", "The interpolation order between levels", "SNESSRMGSetInterpolationOrder", sr->interpOrder, &sr->interpOrder, NULL);CHKERRQ(ierr);
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
  sr->inject      = PETSC_TRUE;
  sr->interpOrder = 1;

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
