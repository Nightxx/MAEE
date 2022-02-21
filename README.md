# MAEE

Simulation code for experiments in the paper: Multi-Aspects Joint Optimization in Recommendation Systems: A Mixture of Aspect-Explicit-Experts Architecture (submitted to KDD 2022).

Environment: Python 3.7.3

## Data Preparation:

### Single-task learning: 
Please download the MovieLens 1M dataset at
```
https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz
```

```
https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.test.gz
```
Then add the two dataset census-income.data.gz and census-income.test.gz to MAEE/MultiTask/data and MAEE/Visualization_MultiTask/data

### Multi-task learning:
Please download the Census Income dataset at
```
https://files.grouplens.org/datasets/movielens/ml-1m.zip
```

Then unzip the package and add the three dataset users.dat, movies.dat, and ratings.dat to MAEE/SingleTask/data and MAEE/Visualization_SingleTask/data




## Experiment Details:

### Single-task learning


+ Comparison of MAEE against DNN, MOE
  
+ Visualization of expert representations
  
+ Rationale analaysis of the masked attention mechanism
  



### Multi-task learning

+ Comparison of MAEE against MMOE, PLE(CGC)

+ Visualization of expert representations
  
