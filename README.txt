README.txt

This file contains information regarding the following files:
	- data_generator.py
	- baseline_fifo_reward.py
	- baseline_random_reward.py
	- submission.py
	- util.py
	- create_graph.py
	- qStarMemoryLog.csv

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

data_generator.py

	This file contains code allowing us to generate data, based on our data collection from Town and Country
	Tennis. To generate data, this file considers the number of hours per day to accept requests, the number
	of days to generate data over, the starting date, and whether or not the Stanford Men's Tennis team is 
	in-season (which impacts the frequency of certain types of requests). You can either accept the default 
	parameters, which are:

						DEFAULT_GENERATE_NUMHOURS = 9
						DEFAULT_NUMDAYS = 6
						DEFAULT_STARTDATE = (1, 1, 2019)
						DEFAULT_IN_SEASON = True,

	or, you may elect to enter your own parameters. To run data_generator.py with your own parameters, 
	execute a command-line call in the format:

	  python3 data_generator.py int(HOURSPERDAY) int(NUMDAYS) str(MM/DD/YEAR) str('True' or 'False')

	The data will then written into a csv file with the following format

		Request ID 		Stanford Men's Tennis	Service Requested			Date 			Time
		int(RequestID)	str('True', 'False')	str('Std', 'Spd', 'Exp')	str(YYYYMMDD)	str(Time HHMM)

	The data will be written to a csv file titled 'training_data_DATETIME', which you can rename manually
	by accessing the csv file in the folder.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

baseline_fifo_reward.py

	This file contains code which executes a FIFO ordering baseline, which reads in a csv file containing 
	our data, and selects racquets to string in a FIFO order of when they were received. To run 
	baseline_fifo_reward.py, execute a command-line call in the format:

			python3 baseline_fifo_reward.py

	To change parameters such as maximum requests per day or the bound of extra racquets to consider per day,
	do so at the top of the main() function.

	OUTPUT: The results printed include the total cumulative racquets completed, through each day, and the
	corresponding reward.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

baseline_random_reward.py

	This file operates in a similar manner to baseline_fifo_received.py, except the racquet-choosing is random 
	rather than in a FIFO ordering.
	
	OUTPUT: The results printed include the total cumulative racquets completed, through each day, and the
	corresponding reward. 

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

submission.py

	This file contains our implementations for Q-Learning. This file contains a RacquetsMDP class to define 
	the MDP for the racquet stringing problem, our QLearningAlgorithm class, and code to test QLearning output
	against Value Iteration (implemented in util.py). Note that due to the slow nature of Value Iteration, this
	should not be tested on large data sets.

		RacquetsMDP
			This class defines the MDP for the racquet stringing model, with the following functions, explained in 
			more detail within the RacquetsMDP class:
						readFile(self, file)
						startState(self)
						actions(self, state)
						succAndProbReward(self, state, action, bounded=(boolean), bound=(int))
						discount(self)

		QLearningAlgorithm
			This class defines important QLearning components which are used in the call to util.simulate(), which
			simulates the QLearning process over the current MDP over a specified number of episodes. The following 
			functions are contained (and explained in more detail) in the QLearningAlgorithm class:
						getQ(self, state, action)
						getAction(self, state)
						getStepSize(self)
						incorporateFeedback(self, state, action, reward, newState)
						updateExplorationProb(self, trialNum, totalTrials)


		testValueIteration(mdp)
			This function simply initializes a ValueIteration class (implemented in util.py) as valueIter, and then
			simulates it with a call to valueIter.solve(mdp, errorTolerance=(float)). Note that Value Iteration is 
			guaranteed to find the policy which returns optimal rewards, but becomes exponentially slower as the number
			of possible states increases, since it must search over every single possible state/action combination at any
			given point in time.

		testQLearning(mdp, printPolicy=(boolean))
			This function simply initializes a QLearning class as qLearn, and then simulates it with a call to
			util.simulate(mdp, qLearn, int(numEpisodes)).

		compareResults(valueIter, qLearn)
			This function compares the results of value iteration and QLeraning with respect to the policies that it produces.
			Note that value iteration is only a viable option for small data sets.


* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

util.py

	This file contains code for ValueIteration, and our simulate function which enables us to run multiple episodes of 
	QLearning. Within the simulate option, you have the option to set 'writeData=True', which will write the reward over
	each episode into a file in a graphable format.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

create_graph.py

	This file contains code allowing us to graph data produced by the simulate funciton in util.py to provide a visual representation
	of our QLearning Algorithm as it improves and approaches a point of convergence.

* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

qStarMemoryLog.csv

	This file serves as a "Memory Log" which allows users to lookup the optimal action from a given state. We continuously
	update qStarMemoryLog.csv with every call to qLearn.simulate, which updates the optimal policy log with new optimal state-
	action pairs learned from the simulation. Furthermore, this memory log also stores corresponding rewards for each optimal state-
	action pair.
