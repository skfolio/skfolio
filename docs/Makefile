# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Cross-platform clean
ifeq ($(OS),Windows_NT)
clean:
	- if exist "$(BUILDDIR)" rmdir /s /q "$(BUILDDIR)"
	- if exist "auto_examples" rmdir /s /q "auto_examples"
	- if exist "generated" rmdir /s /q "generated"
	- if exist "modules" rmdir /s /q "modules"
else
clean:
	-$(RM) -r "$(BUILDDIR)"/* 2>/dev/null || true
	-$(RM) -r auto_examples/* 2>/dev/null || true
	-$(RM) -r generated/* 2>/dev/null || true
	-$(RM) -r modules/* 2>/dev/null || true
endif
