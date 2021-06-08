#!/bin/sh

set -eu

tgt_dir=$(readlink -f $1)
mkdir -pv $tgt_dir

for src in $(find ./ -name "*.tex"); do
    # /path/to/foo/main.tex -> /path/to/foo
    makedir=$(dirname $src)
    make -C $makedir
    # /path/to/foo/main.tex -> tgt_dir/foo.pdf
    cp $makedir/main.pdf $tgt_dir/$(basename $makedir).pdf
done
