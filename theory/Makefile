main=main.tex

all: $(main)
	latexmk $<

preview: $(main)
	latexmk -pvc $<

clean: _restclean
	latexmk -c

allclean: _restclean
	latexmk -C

# Seems like latexmk's "push @generated_exts" doesn't treat directories.
_restclean:
	rm -rf _minted-*
	rm -f *-SAVE-ERROR
