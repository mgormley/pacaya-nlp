# Pacaya NLP [![Build Status](https://travis-ci.org/mgormley/pacaya-nlp.svg?branch=master)](https://travis-ci.org/mgormley/pacaya-nlp)

## Summary 

Pacaya NLP is a suite of Natural Language Processing (NLP) tools built using [Pacaya](https://github.com/mgormley/pacaya), a library for hybrid graphical models and neural networks.

## Using the Library

The latest version is deployed on Maven Central:

```xml
<dependency>
    <groupId>edu.jhu.pacaya-nlp</groupId>
    <artifactId>pacaya-nlp</artifactId>
    <version>3.1.3</version>
</dependency>
```

## Development

### Dependencies

This project has several dependencies all of which are available on Maven Central.
Among others we make extensive use:

* [Prim](https://github.com/mgormley/prim): a Java primitives library
* [Optimize](https://github.com/minyans/optimize): a numerical optimization library
* [Pacaya](https://github.com/mgormley/pacaya): modeling library for hybrids of graphical models and neural networks

#### Installing dependencies locally

1. Checkout and install Prim locally

	```bash
   git clone https://github.com/mgormley/prim.git
	cd prim
	mvn install -DskipTests
	```	 

2. Checkout and install Optimize locally

	```bash
	git clone https://github.com/minyans/optimize.git
	cd optimize
	mvn install -DskipTests
	```

3. Checkout and install Pacaya locally

	```bash
	git clone https://github.com/mgormley/pacaya.git
	cd pacaya
	mvn install -DskipTests
	```
	
### Build:

* Compile the code from the command line:

        mvn compile

* To build a single jar with all the dependencies included:

        mvn compile assembly:single

* To set the classpath using maven:
	
	    source setupenv.sh

### Eclipse setup:

* Create local versions of the .project and .classpath files for Eclipse:

        mvn eclipse:eclipse

* Add M2\_REPO environment variable to
  Eclipse. http://maven.apache.org/guides/mini/guide-ide-eclipse.html
  Open the Preferences and navigate to 'Java --> Build Path -->
  Classpath Variables'. Add a new classpath variable M2\_REPO with the
  path to your local repository (e.g. ~/.m2/repository).

* To make the project Git aware, right click on the project and select Team -> Git... 


## Citations

This library includes code for the papers below. Please cite as appropriate.

```bibtex
@article{gormley_approximation-aware_2015,
    author = {Matthew R. Gormley and Mark Dredze and Jason Eisner},
    title = {Approximation-aware Dependency Parsing by Belief Propagation},
    journal = {Transactions of the Association for Computational Linguistics (TACL)},
    year = {2015}
}
```

```bibtex
@inproceedings{gormley_improved_2015,
    author = {Matthew R. Gormley and Mo Yu and Mark Dredze},
    title = {Improved Relation Extraction with Feature-rich Compositional Embedding Model},
    booktitle = {Proceedings of {EMNLP}},
    year = {2015},
}
```

```bibtex
@inproceedings{gormley_low-resource_2014,
    author = {Gormley, Matthew R. and Mitchell, Margaret and {Van Durme}, Benjamin and Dredze, Mark},
    title = {Low-Resource Semantic Role Labeling},
    booktitle = {Proceedings of {ACL}},
    year = {2014},
}
```
