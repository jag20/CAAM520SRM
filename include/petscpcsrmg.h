#if !defined(_PETSCPCSRMG_H)
#define _PETSCPCSRMG_H

#define PCSRMG "srmg"

/*E
  PCSRMGType - How much data is stored by SRMG

  Level: intermediate

.seealso: PCMG
E*/
typedef enum {PC_SRMG_FULLSPACE,PC_SRMG_PATCHSPACE} PCSRMGType;
PETSC_EXTERN const char *const PCSRMGTypes[];

#endif /* _PETSCPCSRMG_H */
