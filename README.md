# Parallel hyperparameter optimization of spiking neural networks

> [Back to main](https://gitlab.cristal.univ-lille.fr/tfirmin/mythesis)

This repository contains hyperlinks to redirect the reader to the source code of each chapter from the thesis [Parallel hyperparameter optimization of spiking neural networks](https://theses.fr/s327519).

The thesis is accessible at (_available once published_):
* [HAL]()

## Chapter 4 - Partition-based global optimization

Chapter 4 describes a generalization of a family of algorithms, based on a hierarchical decomposition of the search space. We introduce Zellij a framework unifying various algorithms from different research fields. The chapter ends with a discussion about a new algorithm based on a decomposition via Latin Hypercubes.

## Content

The compressed file _coco_data.zip_ contains the results from benchmarking of 12 algorithms on [COCO](https://numbbo.github.io/coco/) and intantiated with [Zellij](https://gitlab.cristal.univ-lille.fr/tfirmin/thesis_code_chap6):
* DIRECT-based
    * DIRECT
    * DIRECT-L
    * DIRECT-R
* SOO-based
    * SOO
    * NMSO
    * SOO-BFS
* FDA-based
    * FDA
    * FDA-D
    * FDA-BFS
    * FDA-DBFS
* 12 LHS-based configurations
* Random multi-start ILS
    * 1RndILS
    * 20RndILS
    * 70RndILS
    
## Zellij

First install the frozen thesis version of **Zellij**.

Zellij is the main Python package made for this thesis, including both FBD and HPO algorithms.
The actual version of the thesis was frozen. The documentation is not up-to-date.

> [Github to Zellij](https://github.com/ThomasFirmin/zellij/tree/thesis_freeze)


## Run 

Shell command to run in parallel all algorithms on all functions and dimension on the COCO benchmark, on a distributed environment with [OAR](https://oar.readthedocs.io/en/2.5/) and [GNU parallel](https://www.gnu.org/software/parallel/).

```
$ parallel --slf $OAR_NODEFILE "source .bashrc; cd ./exp_review; python3 ./fda_exp.py --fct {2} --dim {1} --alg {3} --inst {4}" ::: 2 3 5 10 20 40 ::: $(seq 0 1 23) ::: "70g2LI" "70g6LI" "70g10LI" "70g20LI" "70g2MI" "70g6MI" "70g10MI" "70g20MI" "70g2HI^C"70g6HI" "70g10HI" "70g20HI" ::: $(seq 0 1 14) 1> log.out 2> log.err &
```

Shell command for a FDA algorithm on the function 11 in dimension 2.

```
python3 ./fda_exp.py --fct 11 --dim 2 --alg FDA
```

Shell command to analyze the results.

```
python3 -m cocopp ./exdata/DIRECT ./exdata/SOO ./exdata/FDA ./exdata/20* ./exdata/70RndILS/ ./exdata/1RndILS/
```

Non-COCO analysis is available in ``record.ipynb``.

## Authors and acknowledgment
* Author: Thomas Firmin
* Supervisor: El-Ghazali Talbi
* Co-Supervisor: Pierre Boulet

Experiments presented in this work were carried out using the Grid'5000 testbed, supported by a scientific interest group hosted by Inria and including CNRS, RENATER and several Universities as well as other organizations (see \url{https://www.grid5000.fr}).


This work was granted access to the HPC resources of IDRIS under the allocation 2023-AD011014347 made by GENCI.


This work has been supported by the University of Lille, the ANR-20-THIA-0014 program AI\_PhD$@$Lille and the ANR PEPR AI and Numpex. It was also supported by IRCICA(CNRS and Univ. Lille USR-3380).

## License
CeCILL-C