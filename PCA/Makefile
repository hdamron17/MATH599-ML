all: build/Report.pdf

build/Report.pdf: output/dataset2d-restored.png output/dataset3d-restored.png

output/dataset%d-restored.png: data/dataset%d.csv PCA.py output
	./PCA.py --save-prefix=output/ -spd $(shell seq -s, 1 $$(expr $* - 1) | sed 's/,$$//') $<

data/dataset3d.csv: data
	./gendataset.py

build/%.pdf: %.tex build
	latexmk -pdf --shell-escape -silent -outdir=build "$<"
	@touch $@  # Since latexmk only updates if changed

data output build:
	mkdir -p $@
