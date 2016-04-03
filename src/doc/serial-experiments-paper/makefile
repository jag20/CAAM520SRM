PDFLATEX = pdflatex

all:
	TEXINPUTS=${TEXINPUTS}:./tex BIBINPUTS=${BIBINPUTS}:${PETSC_DIR}/src/docs/tex latexmk -pdf siam.tex

arxiv:
	TEXINPUTS=${TEXINPUTS}:./tex BIBINPUTS=${BIBINPUTS}:${PETSC_DIR}/src/docs/tex latexmk -pdf simple.tex

clean:
	rm -f *.aux *.bbl, *.blg *.fdb_latexmk *.fls *.log *.out simple.pdf siam.pdf

.SUFFIXES: .tex .pdf