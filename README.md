# Summary

Natural Language Processing (NLP) tools built using Pacaya.

# Setup

## Install Dependencies

This project has several dependencies. 
* Prim: a Java primitives library
* Optimize: a Java optimization library
* Pacaya: a Java graphical models library

Currently, these are accessible from the COE's internal maven repository, or they 
can be installed locally. You can use the COE maven repository if you are installing a tagged 
release of Pacaya. If you are trying to build a development branch,
you should install the dependencies locally.

### Using the COE's internal maven repository

If you are installing on the COE grid, you can skip the rest of this 
section.

If you are installing somewhere other than the COE grid, set up an ssh 
tunnel to the COE maven repository and update your settings.xml file 
to point to localhost. 

### Installing dependencies locally

Currently you must request permission to access these private
repositories. Email mrg@cs.jhu.edu for access with your GitLab username.

1. Checkout and install Prim locally
	git clone https://gitlab.hltcoe.jhu.edu/mgormley/prim.git
	cd prim
	mvn install -DskipTests
2. Checkout and install Optimize locally
	git clone https://gitlab.hltcoe.jhu.edu/mgormley/optimize.git
	cd optimize
	mvn install -DskipTests
3. Checkout and install Pacaya locally
	git clone https://gitlab.hltcoe.jhu.edu/mgormley/pacaya.git
	cd pacaya
	mvn install -DskipTests

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

* Add M2_REPO environment variable to
  Eclipse. http://maven.apache.org/guides/mini/guide-ide-eclipse.html
  Open the Preferences and navigate to 'Java --> Build Path -->
  Classpath Variables'. Add a new classpath variable M2_REPO with the
  path to your local repository (e.g. ~/.m2/repository).

* To make the project Git aware, right click on the project and select Team -> Git... 

