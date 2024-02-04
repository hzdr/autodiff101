main=main.tex

all: $(main)
	latexmk $<

preview: $(main)
	latexmk -pvc $<

clean: _restclean
	latexmk -c

allclean: _restclean
	latexmk -C

_restclean:
	rm -rf _minted-*
	rm -f *.bak *.bbl *.run.xml *.nav *.snm *.vrb *.synctex.* *.listing *-SAVE-ERROR
