
PUML = $(wildcard *.puml)
PNG = $(PUML:.puml=.png)

URL = http://sourceforge.net/projects/plantuml/files/plantuml.jar/download

all: $(PNG)

.puml.png:
	@[ -f plantuml.jar ] || curl -L $(URL) -o plantuml.jar
	cat $< | java -jar plantuml.jar -pipe > $@

.SUFFIXES:
.SUFFIXES: .puml .png
