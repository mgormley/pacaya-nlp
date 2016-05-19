# Summary

Pacaya NLP is a suite of Natural Language Processing (NLP) tools built using [Pacaya](https://github.com/mgormley/pacaya), a library for hybrid graphical models and neural networks.

# Using the Library

The latest version is deployed on Maven Central:

```xml
<dependency>
    <groupId>edu.jhu.pacaya-nlp</groupId>
    <artifactId>pacaya-nlp</artifactId>
    <version>3.1.3</version>
</dependency>
```

# Development

## Dependencies

This project has several dependencies all of which are available on Maven Central.
Among others we make extensive use:
* Prim: a Java primitives library
* Optimize: a Java optimization library
* Pacaya: a Java graphical models library

### Installing dependencies locally

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
	
## Build:

* Compile the code from the command line:

        mvn compile

* To build a single jar with all the dependencies included:

        mvn compile assembly:single

* To set the classpath using maven:
	
	    source setupenv.sh

## Eclipse setup:

* Create local versions of the .project and .classpath files for Eclipse:

        mvn eclipse:eclipse

* Add M2\_REPO environment variable to
  Eclipse. http://maven.apache.org/guides/mini/guide-ide-eclipse.html
  Open the Preferences and navigate to 'Java --> Build Path -->
  Classpath Variables'. Add a new classpath variable M2\_REPO with the
  path to your local repository (e.g. ~/.m2/repository).

* To make the project Git aware, right click on the project and select Team -> Git... 

