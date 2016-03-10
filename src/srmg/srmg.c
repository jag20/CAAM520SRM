
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

const char *const PCSRMGTypes[] = {"FULLSPACE","PATCHSPACE","PCSRMGType","PC_SRMG_",0};

/*
   Private context (data structure) for the SRMG preconditioner.
*/
typedef struct {
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
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_SRMG"
static PetscErrorCode PCSetUp_SRMG(PC pc)
{
  PC_SRMG      *sr = (PC_SRMG *) pc->data;
  PetscMPIInt    size;
  PC             pccoarse;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPCreate(PetscObjectComm((PetscObject) pc), &sr->kspcoarse);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(sr->kspcoarse, pc->erroriffailure);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(sr->kspcoarse, "mg_coarse_");CHKERRQ(ierr);

  /* coarse solve is (redundant) LU by default; set shifttype NONZERO to avoid annoying zero-pivot in LU preconditioner */
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
  ierr = KSPSetOperators(sr->kspcoarse, pc->mat, pc->pmat);CHKERRQ(ierr);
  /* Create patch */
  /* Create full space, if necessary */
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
    ierr = PetscViewerASCIIPrintf(viewer, "Coarse grid solver ----------------------------------------\n");CHKERRQ(ierr);
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
  sr->fullspace = PETSC_TRUE;
  sr->kspcoarse = NULL;

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
  PCSRMGGetType - Sets the storage strategy for the preconditioner

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

#undef __FUNCT__
#define __FUNCT__ "PCSRMGInitializePackage"
PetscErrorCode PCSRMGInitializePackage()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCRegister(PCSRMG, PCCreate_SRMG);CHKERRQ(ierr);
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
  ierr = PCSRMGInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
