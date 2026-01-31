.PHONY: latex distill clean all slides slides-all

all: latex distill

latex:
	@echo "Building LaTeX output..."
	@bash build-latex.sh

distill:
	@echo "Building distill.pub output..."
	@bash build-distill.sh

# Build all slides
slides-all:
	@echo "Building all lecture slides..."
	@mkdir -p build/slides
	@cd slides && for f in lecture*.tex; do \
		echo "Building $${f}..."; \
		TEXINPUTS=.:../: pdflatex -output-directory=../build/slides -interaction=nonstopmode "$$f" > /dev/null 2>&1 || true; \
		TEXINPUTS=.:../: pdflatex -output-directory=../build/slides -interaction=nonstopmode "$$f" > /dev/null 2>&1; \
	done
	@echo "Slides built in build/slides/"

# Build a specific lecture slide (usage: make slides LECTURE=01)
slides:
	@echo "Building lecture $(LECTURE) slides..."
	@mkdir -p build/slides
	@cd slides && TEXINPUTS=.:../: pdflatex -output-directory=../build/slides -interaction=nonstopmode lecture$(LECTURE)*.tex > /dev/null 2>&1 || true
	@cd slides && TEXINPUTS=.:../: pdflatex -output-directory=../build/slides -interaction=nonstopmode lecture$(LECTURE)*.tex
	@echo "Slides built in build/slides/"

clean:
	@echo "Cleaning build directories..."
	@rm -rf build/

help:
	@echo "Available targets:"
	@echo "  latex      - Build LaTeX PDF output"
	@echo "  distill    - Build distill.pub HTML output"
	@echo "  slides-all - Build all lecture slides"
	@echo "  slides     - Build specific lecture (e.g., make slides LECTURE=01)"
	@echo "  all        - Build both latex and distill outputs"
	@echo "  clean      - Remove build directories"
