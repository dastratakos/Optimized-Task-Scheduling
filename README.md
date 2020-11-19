<!-- PROJECT SHIELDS -->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Apache-2.0 License][license-shield]][license-url]

<div align="center">
    <h2 style="font-size: 50px; font-weight: bold; margin: 0px;">
        Optimizing task scheduling for small businesses using reinforcement learning
    </h2>
    <h2 style="font-size: 30px; margin: 0px;">
    	Dean Stratakos, Kento Perera, and Timothy Sah
    </h2>
    <a href="(mailto:dstratak@stanford.edu,kperera1@stanford.edu,tsah@stanford.edu)">{dstratak, kperera1, tsah}@stanford.edu</a>
    <br />
	A reinforcement learning model for a
	<a href="https://stanford-cs221.github.io/autumn2019/">Stanford CS 221</a>
	final project.
	<br />
    <a href="Report.pdf">
        Read the report
    </a>
    ·
    <a href="https://github.com/dastratakos/Optimized-Task-Scheduling/issues">
        Request Feature
    </a>
</div>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Abstract](#abstract)
  * [Introduction](#introduction)
  * [Poster](#poster)
* [Details](#details)
* [License](#license)
* [Contact](#contact)

<!-- ABOUT THE PROJECT -->
## About the Project

### Abstract

Small businesses in the service industry face an abundance of challenges.
Oftentimes, their success or failure is closely correlated with their ability
(or lack thereof) to efficiently prioritize tasks in order to maximize profits
on their services. Motivated by a problem we see in our everyday lives and
inspired by the success of Markov Decision Process formulations on problems
such as Blackjack, we developed a system that learns to optimize scheduling
through experience. We present an example solution to a problem of prioritizing
tasks with varying rewards and time constraints by translating it into a
learning problem. Through the course of the project, we settled on two main AI
algorithms: value iteration and Q-learning. These algorithms were compared to
our naive baseline algorithms: random and FIFO. Our results showed that the AI
algorithms outperformed the baselines; both of our AI algorithms adopt policies
that are strategic in hindsight, resulting in a much higher total reward when
compared to the naive algorithms. Furthermore, with respect to value iteration,
Q-learning performs comparably and converges exponentially quicker.

### Introduction

To ground our idea while incorporating personal significance, we designed a
system with a specific local business in mind:
[Tennis Town & Country](http://tennistownandcountry.com/). Tennis Town &
Country is a small tennis shop that provides a tennis racquet stringing service
(in addition to selling tennis-related retail merchandise). This shop has a
special deal with the Stanford Varsity Men’s Tennis Team to string all of their
players’ racquets for a reduced labor rate. This poses a challenge to Tennis
Town & Country, as they have to balance their demand from the Stanford Men’s
Tennis Team with their demand from their regular customers. With a large inflow
of racquets daily, it grows increasingly difficult to account for all of the
factors and optimally prioritize stringing orders. For all of these reasons, we
felt that this shop, in particular, would greatly benefit from having an
algorithm to help them optimize labor allocation when stringing racquets. We
created a system to model their stringing service that maximizes the revenue of
stringing racquets and minimizes the implicit costs of missing deadlines.

### Poster

<img src="Poster.pdf" title="Poster" alt="Poster" width="400" />

<!-- CONTACT -->
## Details

Below is information regarding the following files:
- [`data_generator.py`](#data_generator.py)
- [`baseline_fifo_reward.py`](#baseline_fifo_reward.py)
- [`baseline_random_reward.py`](#baseline_random_reward.py)
- [`model.py`](#model.py)
- [`util.py`](#util.py)
- [`create_graph.py`](#create_graph.py)
- [`qStarMemoryLog.csv`](#qStarMemoryLog.csv)

---

#### `data_generator.py`

This file contains code allowing us to generate data, based on our data
collection from Town and Country Tennis. To generate data, this file considers
the number of hours per day to accept requests, the number of days to generate
data over, the starting date, and whether or not the Stanford Men's Tennis team
is in-season (which impacts the frequency of certain types of requests). You
can either accept the default parameters, which are:

- `DEFAULT_GENERATE_NUMHOURS = 9`
- `DEFAULT_NUMDAYS = 6`
- `DEFAULT_STARTDATE = (1, 1, 2019)`
- `DEFAULT_IN_SEASON = True`,

or, you may elect to enter your own parameters. To run `data_generator.py` with
your own parameters, execute a command-line call in the format:

```sh
python3 data_generator.py int(HOURSPERDAY) int(NUMDAYS) str(MM/DD/YEAR) str('True' or 'False')
```

The data will then written into a csv file with the following format

| Request ID       | Stanford Men's Tennis  |  Service Requested         | Date            | Time             |
|------------------|------------------------|----------------------------|-----------------|------------------|
| `int(RequestID)` | `str('True', 'False')` | `str('Std', 'Spd', 'Exp')` | `str(YYYYMMDD)` | `str(Time HHMM)` |

The data will be written to a csv file titled `data/training_data_DATETIME`,
which you can rename manually by accessing the csv file in the folder.

---

#### `baseline_fifo_reward.py`

This file contains code which executes a FIFO ordering baseline, which reads in
a csv file containing our data, and selects racquets to string in a FIFO order
of when they were received. To run `baseline_fifo_reward.py`, execute a
command-line call in the format:

```sh
python3 baseline_fifo_reward.py
```

To change parameters such as maximum requests per day or the bound of extra
racquets to consider per day, do so at the top of the `main()` function.

OUTPUT: The results printed include the total cumulative racquets completed,
through each day, and the corresponding reward.

---

#### `baseline_random_reward.py`

This file operates in a similar manner to `baseline_fifo_received.py`, except
the racquet-choosing is random rather than in a FIFO ordering.

OUTPUT: The results printed include the total cumulative racquets
completed, through each day, and the corresponding reward. 

---

#### `model.py`

This file contains our implementations for Q-Learning. This file contains a
`RacquetsMDP` class to define the MDP for the racquet stringing problem, our
`QLearningAlgorithm` class, and code to test QLearning output against Value
Iteration (implemented in `util.py`). Note that due to the slow nature of Value
Iteration, this should not be tested on large data sets.

`RacquetsMDP`

This class defines the MDP for the racquet stringing model, with the following
functions, explained in more detail within the `RacquetsMDP` class:
- `readFile(self, file)`
- `startState(self)`
- `actions(self, state)`
- `succAndProbReward(self, state, action, bounded=(boolean), bound=(int))`
- `discount(self)`

`QLearningAlgorithm`

This class defines important QLearning components which are used in the call to
`util.simulate()`, which simulates the QLearning process over the current MDP
over a specified number of episodes. The following functions are contained (and
explained in more detail) in the QLearningAlgorithm class:
- `getQ(self, state, action)`
- `getAction(self, state)`
- `getStepSize(self)`
- `incorporateFeedback(self, state, action, reward, newState)`
- `updateExplorationProb(self, trialNum, totalTrials)`

`testValueIteration(mdp)`
This function simply initializes a `ValueIteration` class (implemented in
`util.py`) as valueIter, and then simulates it with a call to
`valueIter.solve(mdp, errorTolerance=(float))`. Note that Value Iteration is
guaranteed to find the policy which returns optimal rewards, but becomes
exponentially slower as the number of possible states increases, since it must
search over every single possible state/action combination at any given point
in time.

`testQLearning(mdp, printPolicy=(boolean))`

This function simply initializes a QLearning class as qLearn, and then
simulates it with a call to `util.simulate(mdp, qLearn, int(numEpisodes))`.

`compareResults(valueIter, qLearn)`

This function compares the results of value iteration and QLeraning with
respect to the policies that it produces. Note that value iteration is only a
viable option for small data sets.

---

`util.py`

This file contains code for `ValueIteration`, and our simulate function which
enables us to run multiple episodes of QLearning. Within the simulate option,
you have the option to set `writeData=True`, which will write the reward over
each episode into a file in a graphable format.

---

`create_graph.py`

This file contains code allowing us to graph data produced by the `simulate()`
function in `util.py` to provide a visual representation of our QLearning
Algorithm as it improves and approaches a point of convergence.

---

`qStarMemoryLog.csv`

This file serves as a "Memory Log" which allows users to lookup the optimal
action from a given state. We continuously update qStarMemoryLog.csv with every
call to `qLearn.simulate`, which updates the optimal policy log with new
optimal state-action pairs learned from the simulation. Furthermore, this
memory log also stores corresponding rewards for each optimal state-action
pair.

<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See the [`LICENSE`](LICENSE) for more
information.

<!-- CONTACT -->
## Contact

Dean Stratakos, Kento Perera, and Timothy Sah -
[{dstatak, kperera1, tsah}@stanford.edu](mailto:dstratak@stanford.edu,kperera1@stanford.edu,tsah@stanford.edu)

Dean Stratakos: [![LinkedIn][linkedin-shield]][linkedin-url]

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/dastratakos/Optimized-Task-Scheduling.svg?style=flat-square
[contributors-url]: https://github.com/dastratakos/Optimized-Task-Scheduling/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dastratakos/Optimized-Task-Scheduling.svg?style=flat-square
[forks-url]: https://github.com/dastratakos/Optimized-Task-Scheduling/network/members
[stars-shield]: https://img.shields.io/github/stars/dastratakos/Optimized-Task-Scheduling.svg?style=flat-square
[stars-url]: https://github.com/dastratakos/Optimized-Task-Scheduling/stargazers
[issues-shield]: https://img.shields.io/github/issues/dastratakos/Optimized-Task-Scheduling.svg?style=flat-square
[issues-url]: https://github.com/dastratakos/Optimized-Task-Scheduling/issues
[license-shield]: https://img.shields.io/github/license/dastratakos/Optimized-Task-Scheduling.svg?style=flat-square
[license-url]: https://github.com/dastratakos/Optimized-Task-Scheduling/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/dean-stratakos-8b338b149