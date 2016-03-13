
/*  --------------------------------------------------------------------

     This file will implement the SRMG preconditioner in PETSc as part of PC.

     The following basic routines are required for each preconditioner.
          PCCreate_XXX()          - Creates a preconditioner context
          PCSetFromOptions_XXX()  - Sets runtime options
          PCApply_XXX()           - Applies the preconditioner
          PCDestroy_XXX()         - Destroys the preconditioner context
     where the suffix "_XXX" denotes a particular implementation, in
     this case we use _SRMG (e.g., PCCreate_SRMG, PCApply_SRMG).
     These routines are actually called via the common user interface
     routines PCCreate(), PCSetFromOptions(), PCApply(), and PCDestroy(),
     so the application code interface remains identical for all
     preconditioners.

     Another key routine is:
          PCSetUp_XXX()           - Prepares for the use of a preconditioner
     by setting data structures and options.   The interface routine PCSetUp()
     is not usually called directly by the user, but instead is called by
     PCApply() if necessary.

     Additional basic routines are:
          PCView_XXX()            - Prints details of runtime options that
                                    have actually been used.
     These are called by application codes via the interface routines
     PCView().

     The various types of solvers (preconditioners, Krylov subspace methods,
     nonlinear solvers, timesteppers) are all organized similarly, so the
     above description applies to these categories also.  One exception is
     that the analogues of PCApply() for these components are KSPSolve(),
     SNESSolve(), and TSSolve().

     Additional optional functionality unique to preconditioners is left and
     right symmetric preconditioner application via PCApplySymmetricLeft()
     and PCApplySymmetricRight().

    -------------------------------------------------------------------- */

/*
   Include files needed for the SRMG preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <petscpcsrmg.h>
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

const char *const PCSRMGTypes[] = {"FULLSPACE","PATCHSPACE","PCSRMGType","PC_SRMG_",0};

/*
   Private context (data structure) for the SRMG preconditioner.
*/
typedef struct {
  PetscBool setfromopts;
  PetscBool fullspace;
  KSP       kspcoarse;
} PC_SRMG;

#undef __FUNCT__
#define __FUNCT__ "PCSRMGSetType_SRMG"
static PetscErrorCode PCSRMGSetType_SRMG(PC pc, PCSRMGType type)
{
  PC_SRMG *sr = (PC_SRMG *) pc->data;

  PetscFunctionBegin;
  if (type == PC_SRMG_FULLSPACE) {sr->fullspace = PETSC_TRUE;}
  else sr->fullspace = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSRMGGetType_SRMG"
static PetscErrorCode PCSRMGGetType_SRMG(PC pc, PCSRMGType *type)
{
  PC_SRMG *sr = (PC_SRMG *) pc->data;

  PetscFunctionBegin;
  if (sr->fullspace) {*type = PC_SRMG_FULLSPACE;}
  else               {*type = PC_SRMG_PATCHSPACE;}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSRMGCreatePatch_Static"
static PetscErrorCode PCSRMGCreatePatch_Static(DM dm, PetscInt quadrant, PetscInt buffer, DM *patch)
{
  DMDAStencilType stencil_type;
  Vec             gv, lv;
  VecScatter      gtol;
  IS              is;
  PetscInt        dim, dof, s, N, *idx, i;
  PetscInt        xs, xm, ys, ym, zs, zm;
  PetscInt        pxs, pxm, pys, pym, pzs, pzm;
  PetscInt        Xs, Xm, Ys, Ym, Zs, Zm;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  if (buffer) SETERRQ1(comm, PETSC_ERR_SUP, "Only support buffer size 0, not %d", buffer);
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
  ierr = DMDAGetCorners(dm, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dm, &Xs, &Ys, &Zs, &Xm, &Ym, &Zm);CHKERRQ(ierr);
#if 0
  if (quadrant & 0x1) {pxs = xs + xm/2; pxm = xm - xm/2;}
  else                {pxs = xs;        pxm = xm/2;}
  if (quadrant & 0x3) {pys = ys + ym/2; pym = ym - ym/2;}
  else                {pys = ys;        pym = ym/2;}
  if (quadrant & 0x5) {pzs = zs + zm/2; pzm = zm - zm/2;}
  else                {pzs = zs;        pzm = zm/2;}
#else
  pxs = xs; pxm = xm;
  pys = ys; pym = ym;
  pzs = zs; pzm = zm;
#endif
  ierr = DMDASetSizes(*patch, pxm, pym, pzm);CHKERRQ(ierr);
  ierr = DMDASetCorners(*patch, xs, ys, zs, xm, ym, zm);CHKERRQ(ierr);
  ierr = DMDASetGhostCorners(*patch, xs, ys, zs, xm, ym, zm);CHKERRQ(ierr);
  /* TODO Project coordinates just like solution */
  ierr = DMDASetUniformCoordinates(*patch, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMCopyDMSNES(dm, *patch);CHKERRQ(ierr);
  /* Now I have to do the rest of DMSetUp() since the DA version does too much we do not want */
  {
    DM_DA *da = (DM_DA *) (*patch)->data;

    da->Nlocal = xm*ym*zm*dof;
    da->nlocal = xm*xm*zm*dof; /* TODO Should be ghosted sizes */
  }
  /*   Create scatter and l2g map, which are useless here */
  N    = pxm*pym*pzm;
  ierr = PetscMalloc1(N, &idx);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) idx[i] = i;
  ierr = ISCreateBlock(PETSC_COMM_SELF, dof, N, idx, PETSC_OWN_POINTER, &is);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(is, &(*patch)->ltogmap);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject) *patch, (PetscObject) (*patch)->ltogmap);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(*patch, &gv);CHKERRQ(ierr);
  ierr = DMGetLocalVector(*patch, &lv);CHKERRQ(ierr);
  ierr = VecScatterCreate(gv, is, lv, is, &gtol);CHKERRQ(ierr);
  ierr = DMDASetScatter(*patch, gtol, NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject) *patch, (PetscObject) gtol);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(*patch, &gv);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(*patch, &lv);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  dm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_SRMG - Prepares for the use of the SRMG preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.

   pcsetupcalled = 0 means that PCSetUp() has never been called
   pcsetupcalled = 1 means that PCSetUp() has been called before
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_SRMG"
static PetscErrorCode PCSetUp_SRMG(PC pc)
{
  PC_SRMG       *sr = (PC_SRMG *) pc->data;
  PetscMPIInt    size;
  PC             pccoarse;
  PetscInt       tab;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    /* Create coarse solver */
    ierr = KSPCreate(PetscObjectComm((PetscObject) pc), &sr->kspcoarse);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(sr->kspcoarse, pc->erroriffailure);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(sr->kspcoarse, "srmg_coarse_");CHKERRQ(ierr);
    ierr = PetscObjectGetTabLevel((PetscObject) pc, &tab);CHKERRQ(ierr);
    ierr = KSPSetTabLevel(sr->kspcoarse, tab+1);CHKERRQ(ierr);
    /*   coarse solve is (redundant) LU by default; set shifttype NONZERO to avoid annoying zero-pivot in LU preconditioner */
    ierr = KSPSetType(sr->kspcoarse, KSPPREONLY);CHKERRQ(ierr);
    ierr = KSPGetPC(sr->kspcoarse, &pccoarse);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject) pc), &size);CHKERRQ(ierr);
    if (size > 1) {
      KSP innerksp;
      PC  innerpc;

      ierr = PCSetType(pccoarse, PCREDUNDANT);CHKERRQ(ierr);
      ierr = PCRedundantGetKSP(pccoarse, &innerksp);CHKERRQ(ierr);
      ierr = KSPGetPC(innerksp, &innerpc);CHKERRQ(ierr);
      ierr = PCFactorSetShiftType(innerpc, MAT_SHIFT_INBLOCKS);CHKERRQ(ierr);
    } else {
      ierr = PCSetType(pccoarse, PCLU);CHKERRQ(ierr);
      ierr = PCFactorSetShiftType(pccoarse, MAT_SHIFT_INBLOCKS);CHKERRQ(ierr);
    }
  }
  ierr = KSPSetOperators(sr->kspcoarse, pc->mat, pc->pmat);CHKERRQ(ierr);
  if (sr->setfromopts) {ierr = KSPSetFromOptions(sr->kspcoarse);CHKERRQ(ierr);}
  ierr = KSPSetUp(sr->kspcoarse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  PCApply_SRMG - Applies the SRMG preconditioner to a vector.

  Input Parameters:
+ pc - the preconditioner context
- x - input vector

  Output Parameter:
. y - output vector
*/
#undef __FUNCT__
#define __FUNCT__ "PCApply_SRMG"
static PetscErrorCode PCApply_SRMG(PC pc, Vec x, Vec y)
{
  PC_SRMG       *sr = (PC_SRMG *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPSolve(sr->kspcoarse, x, y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_SRMG"
static PetscErrorCode PCReset_SRMG(PC pc)
{
  PC_SRMG       *sr = (PC_SRMG *) pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sr->setfromopts = PETSC_FALSE;
  /* Destroy structures */
  ierr = KSPDestroy(&sr->kspcoarse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  PCDestroy_SRMG - Destroys the private context for the SRMG preconditioner that was created with PCCreate_SRMG().

  Input Parameter:
. pc - the preconditioner context
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_SRMG"
static PetscErrorCode PCDestroy_SRMG(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_SRMG(pc);CHKERRQ(ierr);

  /* Free the private data structure that was hanging off the PC */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_SRMG"
static PetscErrorCode PCSetFromOptions_SRMG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_SRMG       *sr = (PC_SRMG *) pc->data;
  PCSRMGType     deftype, type;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sr->setfromopts = PETSC_TRUE;
  ierr = PCSRMGGetType(pc, &deftype);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject, "SRMG options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_srmg_type", "How much to store", "PCSRMGSetType", PCSRMGTypes, (PetscEnum) deftype, (PetscEnum *) &type, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PCSRMGSetType(pc, type);CHKERRQ(ierr);}
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_SRMG"
PetscErrorCode PCView_SRMG(PC pc, PetscViewer viewer)
{
  PC_SRMG       *sr = (PC_SRMG *) pc->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    if (sr->fullspace) {ierr = PetscViewerASCIIPrintf(viewer,"  SRMG storing the full space\n");CHKERRQ(ierr);}
    else               {ierr = PetscViewerASCIIPrintf(viewer,"  SRMG storing only the patch space\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "SRMG Coarse grid solver ----------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = KSPView(sr->kspcoarse, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCCreate_SRMG - Creates an SRMG preconditioner context, PC_SRMG,
   and sets this as the private data within the generic preconditioning
   context, PC, that was created within PCCreate().

   Input Parameter:
.  pc - the preconditioner context
*/

/*MC
  PCSRMG - Segmental Refinement Multigrid (i.e. low-memory preconditioning)

  Options Database Key:
. -pc_srmg_type <fullspace,patchspace> - decide how much of the solution to store

  Level: beginner

  Concepts: segmental refinement, multigrid, preconditioners

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSRMGSetType()
M*/
#undef __FUNCT__
#define __FUNCT__ "PCCreate_SRMG"
PETSC_EXTERN PetscErrorCode PCCreate_SRMG(PC pc)
{
  PC_SRMG       *sr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr     = PetscNewLog(pc, &sr);CHKERRQ(ierr);
  pc->data = (void *) sr;

  /*
     Initialize the pointers to vectors to ZERO; these will be used to store
     diagonal entries of the matrix for fast preconditioner application.
  */
  sr->fullspace   = PETSC_TRUE;
  sr->setfromopts = PETSC_FALSE;
  sr->kspcoarse   = NULL;

  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_SRMG;
  pc->ops->applytranspose      = PCApply_SRMG;
  pc->ops->setup               = PCSetUp_SRMG;
  pc->ops->reset               = PCReset_SRMG;
  pc->ops->destroy             = PCDestroy_SRMG;
  pc->ops->setfromoptions      = PCSetFromOptions_SRMG;
  pc->ops->view                = PCView_SRMG;
  pc->ops->applyrichardson     = 0;

  ierr = PetscObjectComposeFunction((PetscObject) pc, "PCSRMGSetType_C", PCSRMGSetType_SRMG);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) pc, "PCSRMGGetType_C", PCSRMGGetType_SRMG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSRMGSetType"
/*@
  PCSRMGSetType - Sets the storage strategy for the preconditioner

  Logically Collective on PC

  Input Parameters:
+ pc - the preconditioner context
- type - PC_SRMG_FULLSPACE, PC_SRMG_PATCHSPACE

  Options Database Key:
. -pc_srmg_type <fullspace,patchspace>

  Level: intermediate

  Concepts: SRMG preconditioner

.seealso: PCSRMGGetType()
@*/
PetscErrorCode PCSRMGSetType(PC pc, PCSRMGType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscTryMethod(pc, "PCSRMGSetType_C", (PC,PCSRMGType), (pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSRMGGetType"
/*@
  PCSRMGGetType - Gets the storage strategy for the preconditioner

  Not Collective on PC

  Input Parameter:
. pc - the preconditioner context

  Output Parameter:
. type - PC_SRMG_FULLSPACE, PC_SRMG_PATCHSPACE

  Level: intermediate

  Concepts: SRMG preconditioner

.seealso: PCSRMGSetType()
@*/
PetscErrorCode PCSRMGGetType(PC pc, PCSRMGType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  ierr = PetscUseMethod(pc, "PCSRMGGetType_C", (PC,PCSRMGType*), (pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------- SNES ------------------------------------- */

typedef struct {
  PetscBool setfromopts;
  PetscBool fullspace;
  SNES      solCoarse;
  SNES      solPatch;
  PetscInt  numLevels;
  DM       *patches;
} SNES_SRMG;

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
#define __FUNCT__ "SNESSolve_SRMG"
static PetscErrorCode SNESSolve_SRMG(SNES snes)
{
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  Vec            oldsol;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (snes->xl || snes->xu || snes->ops->computevariablebounds) SETERRQ1(PetscObjectComm((PetscObject) snes), PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject) snes)->type_name);
  ierr = PetscCitationsRegister(SRMGCitation, &SRMGcite);CHKERRQ(ierr);

  ierr = VecDuplicate(snes->vec_sol, &oldsol);CHKERRQ(ierr);
  ierr = VecCopy(snes->vec_sol, oldsol);CHKERRQ(ierr);
  ierr = SNESSolve(sr->solCoarse, snes->vec_rhs, snes->vec_sol);CHKERRQ(ierr);
  {
    DM dm, dmPatch;

    ierr = SNESReset(sr->solPatch);CHKERRQ(ierr);
    ierr = SNESGetDM(sr->solCoarse, &dm);CHKERRQ(ierr);
    /* TODO Must interpolate coordinates */
    ierr = PCSRMGCreatePatch_Static(dm, 0, 0, &dmPatch);CHKERRQ(ierr);
    ierr = SNESSetDM(sr->solPatch, dmPatch);CHKERRQ(ierr);
    ierr = DMDestroy(&dmPatch);CHKERRQ(ierr);
    if (sr->setfromopts) {ierr = SNESSetFromOptions(sr->solPatch);CHKERRQ(ierr);}
    /* TODO Must interpolate rhs and solution vector */
    /* FORNOW Reset solution so I can check that patch works */
    ierr = VecCopy(oldsol, snes->vec_sol);CHKERRQ(ierr);
    ierr = SNESSolve(sr->solPatch, snes->vec_rhs, snes->vec_sol);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&oldsol);CHKERRQ(ierr);
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
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  sr->setfromopts = PETSC_TRUE;
  ierr = PetscOptionsHead(PetscOptionsObject, "SRMG options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_srmg_levels", "How many refinements to make", "SNESSRMGSetNumLevels", sr->numLevels, &sr->numLevels, NULL);CHKERRQ(ierr);
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
    if (sr->fullspace) {ierr = PetscViewerASCIIPrintf(viewer,"  SRMG with %D levels, storing the full space\n", sr->numLevels);CHKERRQ(ierr);}
    else               {ierr = PetscViewerASCIIPrintf(viewer,"  SRMG with %D levels, storing only the patch space\n", sr->numLevels);CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "SRMG Coarse grid solver ----------------------------------------\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = SNESView(sr->solCoarse, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
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

  sr->fullspace   = PETSC_TRUE;
  sr->setfromopts = PETSC_FALSE;
  sr->solCoarse   = NULL;
  sr->numLevels   = 0;
  sr->patches     = NULL;

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
  ierr = PCRegister(PCSRMG, PCCreate_SRMG);CHKERRQ(ierr);
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
  SNES_SRMG     *sr = (SNES_SRMG *) snes->data;
  PetscErrorCode ierr;

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
. n - the nubmer of levels

  Level: intermediate

  Concepts: SRMG preconditioner

.seealso: SNESSRMGSetNumLevels()
@*/
PetscErrorCode SNESSRMGGetNumLevels(SNES snes, PetscInt *n)
{
  SNES_SRMG       *sr = (SNES_SRMG *) snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidPointer(n, 2);
  *n = sr->numLevels;
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
