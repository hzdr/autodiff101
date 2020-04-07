all:
	# --shell-escape b/c of minted package
	latexmk -pdf -pvc -pdflatex="pdflatex --shell-escape %O %S" theory.tex

clean: _restclean
	latexmk -c

allclean: _restclean
	latexmk -C

_restclean:
	rm -rf _minted-*
	rm -f *.bak
