---
bibliography:
- 'IEEEabrv.bib'
- 'extra.bib'
title: |
    Tsetlin Machines using Deterministic Reinforcement and Hyper-parameter
    Search
katex: true
markup: "mmark"
---

Introduction
============

The Tsetlin Machine (TM) [@DBLP:journals/corr/abs-1804-01508] is a novel
Machine Learning (ML) architecture which is particularly suited to low
power and explainable applications. Since the algorithm itself is built
using existing logic building blocks, there is the potential to build
learning systems at the edge with a fraction of the hardware utilised by
existing Deep Learning (DL) methods, with the same level of performance.
This paper describes an initial practical assessment, which deliberately
focuses on the area of hyper-parameter tuning and use of randomness
which can be particularly costly energy wise.

The paper is constructed as follows, firstly, we describe our
interpretation of the TM algorithm, with reference to the learning
procedure presented in the originating paper. We then explain
modifications to the algorithm to test aspects of the initialisation
particularly related to controlling the amount of randomness. In this
section we describe an approximation, which takes the further step of
removing some of the necessity for randomness. We test this together
with hyper-parameter selection on three well known classification
data-sets, using cross-validated test set accuracy. Finally, we discuss
the results and future work.

Algorithm Description
=====================

The underlying structure of the TM is the same as the majority of
learning classification algorithms with forward pass inference and a
backward pass parameter adjusting learning step. A key feature of the
algorithm is that the whole system is Boolean, requiring an encoder and
decoder (explained more in Figure [\[fig2\]](#fig2){reference-type="ref"
reference="fig2"}) for any other form of data. Data is defined as the
set of vectors:

$$ D_n= \{x_1,x_2,...,x_n\}$$

And output is:

$$Y = f(x;T)$$

Where Y is a "class" output $$(Y_{Cl}= \{y_1,y_2,...,y_{cl}\})$$ The
system is discriminatory ($p(Y|D=x)$) not generative ($p(D|Y=y)$).

The process of learning is the optimisation of Tsetlin automata
parameters $t$ (labelling the full collection $T$) such that the
function provides the most accurate estimate of a given set of classes
$Y$. The inference mechanism is an ensemble of disjunctive clauses, each
clause controlling the inclusion of a feature and (separately) its
negation. Formally a clause ($c_{jcl}$ is the $j$th clause for class
$cl$ with $J$ the total number of classes) is:

$$c_j= \bigwedge\limits^2F_{i=0}(x^i \land a^i) \land (\neg x^{2i} \land a^{2i})$$

Where $x^i$ is the $ith$ feature of vector $x$ and $F$ is the total
number of features and $a^i$ is defined below. A variety of ensembling
methods can be used - most commonly single voted threshold, with half
the clauses aimed at inhibition: $$\label{ex}
y = 1   \quad if  \quad (\sum_{j=0}^{J/2} c_{j0} -\sum_{j=j/2+1}^Jc_{j0} ) \ge 0  \quad  else  \quad 0$$

Where the algorithm is multi-class (one-hot encoded \[15\]) thresholding
is removed and the maximum vote is used to assess the estimated class:
$$y = y_{cl} \; if \; max(\sum_{j=0}^{J/2}c_{jcl} - \sum_{j=(j/2)+1}^Jc_{jcl} ) \quad \forall \;  y_{cl} \;  in \: Y_{Cl}$$

The learning mechanism controls the inclusion of features or their
negation and uses classic Tsetlin automata binary action (a) [@tset],
which exist as thresholdable integer count values. Formally, a Tsetlin
automata ($t^i$) is: $$a^i=1 \: if \: t^i > t_{thres} \: else \: 0$$ A
reward for a Tsetlin automata is $t^i+=1$ whilst a penalty is $t^i-=1$.
The learning rules are described in detail in
[@DBLP:journals/corr/abs-1804-01508], the basic premise being that
individual automata can be controlled by examining their effect on the
clause output relative to the desired output ($y_{est}$), the clause
output, the polarity (an inhibitory or excitory clause) and the action
($a^i$). For example, if a feature is $1$ and the Tsetlin ($t$) is
resolved to action ($a \land x = 1$) and the overall clause is
inhibiting when an 0 output is desired, then the Tsetlin is rewarded
(+1), as it is trying to do the right action.

The learning approach has the following added randomness:

-   An annealing style cooling parameter $Temp$, which we set to $15$ in
    this paper, and have found has little effect.

-   Sensitivity $S$ which controls the ratio of reward to penalty, based
    on a split of the total probability, with values $1/S$ and $1-1/S$.
    This is used in a *choice* function on a tuple $(penalty,reward)$ as
    shown in Table [\[app\]](#app){reference-type="ref" reference="app"}
    in the Appendix.

Again, for further background and justification see
[@DBLP:journals/corr/abs-1804-01508].

Comparisons with MLP based learning
-----------------------------------

It is worthwhile to compare the TMs with a classical MLP [@rum] (see
Figure [\[fig1\]](#fig1){reference-type="ref" reference="fig1"}),
noteworthy points include: Boolean/integer rather than the floating
point internally; logic rather than linear algebra; and binary
inputs/outputs.

![The backwards pass requires no gradient methods (just manipulation of
all Tsetlin states (T)). Due to the negation of features, clauses can
singularly encode non linearly separable boundaries. There are no
sigmoid style functions, all functions are compatible with low level
hardware implementations. "Weights" (Tsetlin Automata) in the TM are
thresholded not "neurons" (clauses). []{label="fig1"}](net.png){#fig1}

TMs and Binary Neural Networks (BNNs)
[@DBLP:journals/corr/CourbariauxB16] share many similarities, most
notably the resultant inference mechanism being purely logic based.
However, the major contrast is that TMs eradicate non-Boolean values
during training by encoding the non-Boolean inputs and decoding at the
output phase, whereas BNNs transform the entire internal representation
after training (see Figure [\[fig2\]](#fig2){reference-type="ref"
reference="fig2"}).

![TMs vs BNN's: A TM (1) carries it's encoding and decoding strategy
into the inference process, where ANN adapts it's internal weight via
quantisation.[]{label="fig2"}](tmnn1.png){#fig2}

Modifications to Randomness
---------------------------

To assess the sensitivity to stochasticity discussed above, we produce a
series of random samples at different cycle lengths - the numbers are
generated from a uniform distribution over a partitioned interval, so,
for a two cycle array random numbers are generated in the range $0-0.49$
and $0.5-1$ and then shuffled. As a further example, for a ten cycle
array random numbers are generated in the random array:
$\{0 < r_1 \leq 0.09,0.1<r_2 \leq 0.19,0.2<r_3 \leq 0.29, 0.3<r_4 \leq 0.39,0.4<r_5 \leq0.49,0.5 
<r_6 \leq 0.59,  0.6<r_7 \leq0.69,0.7<r_8 \leq 0.79, 0.8<r_9 \leq 0.89,$
$0.9 < r_{10} \leq 0.99\}$ which is then shuffled.

Deterministic Learning
----------------------

The algorithm can be further modified to replace the stochastic
expectation with a guaranteed reinforcement signal, replacing long-range
expectations with single update values. We rely on the fact that at
every epoch of training the learning effect can be approximated by the
instantaneous effect of the penalty and reward combined. Mathematically,
if we assume that the final state ($F$) is a sum collection (over a time
period $T$) of updates ($u$) and this is a collection of penalties ($p$)
and rewards ($r$):

$$F \pm \sum^T_{t=0}u_t \approx F \pm\sum^T_{t=0}|r_t-p_t|$$

This removes the requirement for choice between reward and penalty at
the cost of a non 1/-1 based accumulation, with the implication being
float point based Tsetlin Automata units. In practice we have
experimented with reward/penalty combinations that produce integer
values, and will report this in a later paper.

Empirical Evaluation
====================

This section describes how we have assessed the practicalities of this
algorithm, paying particular attention to the practicalities of
establishing the correct hyper-parameters.

Experimental Design
-------------------

We take three data-sets, the classic `Iris` data-set, `Digit` and
`Breast`, and apply best practice 10-fold cross validation over a
selection of hyper-parameters and then report out of sample train/test
set error. For reference, `XGBoost` [@chen2016xgboost] test set
performance is included (implemented via the `sklearn` python library
[@scikit-learn]).

Data-sets
---------

The emphasis of this experimentation is to establish the ability to work
in the same domain as the `XGBoost` algorithm, with a bias towards
smaller samples (rather than DL style large data-sets). The data-sets
are readily available (in `sklearn`) standard machine learning problems.
Table [\[t4\]](#t4){reference-type="ref" reference="t4"} gives the
python function call to access the data. Training was not batched.

Due to the Boolean nature of the algorithm and the ranges of the
features, we transformed any float or integer into a unary "thermometer"
code based on a 5 bit discretisation across the dynamic range (*1 bit
step* = $(max(x^i)-min(x^i))/5$).

  Data-set    No.   Feat/Class             sklearn command
  ---------- ------ ---------------------- ------------------------
  Iris        150   4 `Flt` inputs, 3cls   `load_iris()`
  Breast      569   30 `Flt`, 2cls         `load_breast_cancer()`
  Digit       1797  64 `4bit Int`, 10cls   `load_digits()`

  : Data-sets used in evaluation, **No.** is the number of examples in
  the data-set.[]{label="t4"}

Hyper-parameter Settings
------------------------

The following hyper-parameters exist for the learning algorithm:

  Variable   Settings
  ---------- ----------------------
  RType      $[LR,HR,D]$
  cl         $[10,20,50,100,200]$
  it         $[20,50,100,200]$
  S          $[2,3.9,6]$

  :  *Iterations (it)* - the maximum number of epochs for learning
  update, *Randomisation Type* (RType) *Number of clauses (cl)* - the
  sensitivity parameter (S).

with randomness settings as Table [\[tabr\]](#tabr){reference-type="ref"
reference="tabr"}

  Randomisation (Rtype)    cycle size
  ----------------------- ------------
  Low Random (LR)             100
  High Random (HR)            1000
  Deterministic (D)            \-

  : Random settings.[]{label="tabr"}

Results
=======

Tables [\[t1\]](#t1){reference-type="ref" reference="t1"},
[\[t2\]](#t2){reference-type="ref" reference="t2"} and
[\[t3\]](#t3){reference-type="ref" reference="t3"} present the
experimental results, for brevity we include the top 10 results, which
cover best results for all the algorithm apart from the ranked 15th
result for the **D** on `Digit`.

   RType   Cl    Iter    S    Trn/mean      Trn/std     Tst/mean   Tst/std 
  ------- ----- ------ ----- ---------- ----------- ------------ --------- --
                                          **`XGB`**   **0.9543**     0.022 
     D     100    20    3.9    0.972          0.003        0.961     0.021 
     D     100   100    3.9    0.976          0.005        0.960     0.023 
     D     100    50    3.9    0.974          0.003        0.959     0.024 
    HR     50    200     2     0.968          0.003        0.959     0.024 
     D     100    20     6     0.971          0.004        0.957     0.030 
    LR     100    50    3.9    0.973          0.004        0.957     0.023 
    HR     200   200     6     0.987          0.004        0.955     0.028 
     D     200   200     6     0.988          0.004        0.955     0.026 
    LR     200   200     6     0.988          0.004        0.955     0.034 

  : Top 10 sorted by Test set error - `Iris` data-set results. The
  **`XGB`** row is the **`XGBoost`** results when applied to this
  data-set.[]{label="t1"}

   RType   Cl    Iter    S    Trn/mean      Trn/std    Tst/mean   Tst/std 
  ------- ----- ------ ----- ---------- ----------- ----------- --------- --
                                          **`XGB`**   **0.927**      0.08 
    HR     200   100     6     0.979          0.004      0.9268     0.062 
    LR     100   200    3.9    0.974          0.006      0.9267     0.069 
     D     100    20     6     0.962          0.014        0.92     0.071 
     D     100   100    3.9    0.968          0.005        0.92     0.071 
     D     200   100     6     0.974          0.007        0.92     0.071 
    HR     200   200     6     0.977          0.006        0.92     0.057 
     D     100    50    3.9    0.973          0.006        0.92     0.065 
     D     100    50     6     0.963          0.005        0.92     0.083 
     D     100   100     6     0.961          0.008        0.92     0.083 
     D     100   200    3.9    0.965          0.008        0.92     0.065 

  : Top 10 sorted by Test set error - `Breast` data-set
  results.[]{label="t2"}

     RType     Cl    Iter    S     Trn/mean     Trn/std    Tst/mean   Tst/std
  ----------- ----- ------ ----- ---------- ----------- ----------- ---------
                                              **`XGB`**   **0.966**     0.008
      LR       20     20     2        0.976       0.005       0.959     0.023
      LR       10    200     2        0.982       0.004       0.954     0.020
      HR       10    100     6        0.981       0.006       0.952     0.030
      LR       20     50    3.9       0.967       0.005       0.948     0.027
      LR       20    200     6        0.971       0.005       0.948     0.027
      HR       20     50     2        0.972       0.003       0.948     0.031
      LR       200   100    3.9       0.982       0.005       0.946     0.019
      HR       50     50    3.9       0.982       0.005       0.946     0.019
      HR       50    100     6        0.982       0.005       0.946     0.019
      LR       200   200     6        0.964       0.005       0.945     0.026
   $...15$ D   200   200    3.9       0.989       0.003       0.942     0.024

  : Top 10 sorted by Test set error - `Digit` data-set results. Note
  that the highest deterministic (D) result is include at position 15 in
  the table.[]{label="t3"}

Discussion and Further Work
===========================

The following observations are noteworthy:

-   In general TMs have low variance between training runs - training is
    a fairly stable process given a set of parameters.

-   **`XGBoost`** - , which is excellent, considering that this is
    thought to be the best known \"out-the-box\" classifier.

-   **D** - Surprisingly, the deterministic algorithm features highly in
    two of the three data-sets, with strong performance in the `Breast`
    data. Also, it seems this algorithm requires less training than the
    LR, HR perhaps due to a more predictable reinforcement given the
    learning signal.

-   **LR** - Low Randomness features strongly in the evaluation with
    best performing test-set error in `Digit` and `Breast` data-set
    evaluations.

-   **HR** - High Randomness does not feature as readily as
    anticipated - it is clearly the case that limiting the entropy
    within the learning process does not unduly effect overall test set
    accuracy.

-   **Class size** - In general there was a spread of class sizes,
    surprisingly the `Digit` problem features low class (20) for the
    best performing.

-   **Iterations** - For each data-set, there is always a TM, trained
    over less than 50 iterations, within 2% of `XGBoost` test set
    accuracy. There was also a variety of iteration sizes represented in
    the top ten of each, with no real bias towards higher iterations. It
    appears that the deterministic (D) algorithm can be run for less
    iterations than the random algorithms.

-   ***S*** - Again the full range of $S$ are represented in the best
    results, although there is perhaps a bias towards the training
    cycles using $6$.

In summary, this algorithm, especially in it's low and deterministic
form, offers a viable method of building a classifier, with a stable
simple learning, and unlike BNNs no post training quantisation process.

For future work we are particularly interested in how this might
transfer to hardware, as unlike MLPs the operations are all standard
logic operations and the method can be readily parallelised at the
clause level. Furthermore, the clauses form compact logical
representations that are amenable to further analysis to gain a stronger
understanding of the internalised knowledge. Clearly, if the algorithm
can be successfully applied to the latest "Deep" data-sets, then there
is huge potential to build explainable and fast hardware
implementations.

\appendix
   y   y   polarity   clause   literal x   exc/inc   (re,in,pen)
  --- --- ---------- -------- ----------- --------- -------------
   1   0    0 (-1)      0          0          0        (L,B,0)
   1   0    0 (-1)      0          0          1        (0,B,L)
   1   0    0 (-1)      0          1          0        (L,B,0)
   1   0    0 (-1)      0          1          1        (0,B,L)
   1   0    0 (-1)      1          0          0        (L,B,0)
   1   0    0 (-1)      1          0          1           0
   1   0    0 (-1)      1          1          0        (0,L,B)
   1   0    0 (-1)      1          1          1        (B,L,0)
   1   0      1         0          0          0           0
   1   0      1         0          0          1           0
   1   0      1         0          1          0           0
   1   0      1         0          1          1           0
   1   0      1         1          0          0        (0,0,1)
   1   0      1         1          0          1           0
   1   0      1         1          1          0           0
   1   0      1         1          1          1           0
   0   1    0 (-1)      0          0          0           0
   0   1    0 (-1)      0          0          1           0
   0   1    0 (-1)      0          1          0           0
   0   1    0 (-1)      0          1          1           0
   0   1    0 (-1)      1          0          0        (0,0,1)
   0   1    0 (-1)      1          0          1           0
   0   1    0 (-1)      1          1          0           0
   0   1    0 (-1)      1          1          1           0
   0   1      1         0          0          0        (L,B,0)
   0   1      1         0          0          1        (0,B,L)
   0   1      1         0          1          0        (L,B,0)
   0   1      1         0          1          1        (0,B,L)
   0   1      1         1          0          0        (L,B,0)
   0   1      1         1          0          1           0
   0   1      1         1          1          0        (0,L,B)
   0   1      1         1          1          1        (1,0,0)

  : Update lookup where $L=1/S$ $B= S/S-1$ as per
  [@DBLP:journals/corr/abs-1804-01508], Polarity is whether the clause
  is excitory or inhibitory in Equation
  [\[ex\]](#ex){reference-type="ref" reference="ex"}.

[\[app\]]{#app label="app"}

\bibliographystyle{IEEEtran}
