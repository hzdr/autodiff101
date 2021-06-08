# --shell-escape b/c of minted package
opts="-pdf -pdflatex='pdflatex --shell-escape %O %S'"

all: main.tex
	latexmk "$(opts)" $<

preview: main.tex
	latexmk -pvc "$(opts)" $<

clean: _restclean
	latexmk -c

allclean: _restclean
	latexmk -C

_restclean:
	rm -rf _minted-*
	rm -f *.bak *.bbl *.run.xml *.nav *.snm *.vrb
