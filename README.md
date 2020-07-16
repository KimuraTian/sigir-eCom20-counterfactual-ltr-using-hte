# Counterfactual Learning-to-Rank using Heterogeneous Treatment Effect Estimation

## Requirements:
- Anaconda, scipy, pandas, tensorflow_ranking=0.1.5, tensorflow=1.15.0, scikit-learn, invoke, pytest, seaborn, grf, xgboost, causalml

## Steps to run the experiment.

### 1. Download source data
1. Download data set `MSLR-WEB30K` from https://www.microsoft.com/en-us/research/project/mslr/.
2. Extract data to `/sigir-eCom20-counterfactual-ltr-using-hte/data/`

### 2. Make splits
1. go to the repo main directory (e.g. /projects/sigir-eCom20-counterfactual-ltr-using-hte/)
2. Run shell command: `invoke normalize-and-split-raw-data --data-paths=/PATH_TO_TRAIN_DATA/train.txt --data-paths=/PATH_TO_VALIDATION_DATA/train.txt --data-paths=/PATH_TO_TEST_DATA/train.txt --save-to-dir=/PATH_TO_OUTPUT_FOLDER/`
3. Example command for one fold: `invoke normalize-and-split-raw-data --data-paths=data/MSLR-WEB30K/Fold1/train.txt --data-paths=data/MSLR-WEB30K/Fold1/vali.txt --data-paths=data/MSLR-WEB30K/Fold1/test.txt --save-to-dir=build/simulation/Fold1`

### 3. Train and serve a production ranker
1. Run shell command: `invoke train-and-serve-production-ranker --data-paths=build/simulation/Fold1/sim_prod_ranker_train.txt --data-paths=build/simulation/Fold1/sim_prod_ranker_vali.txt --data-paths=build/simulation/Fold1/sim_prod_ranker_train_vali.txt --tmp-sweep-model-dir=build/simulation/Fold1/sweep --prod-ranker-dir=build/simulation/Fold1/prod_ranker --exp-train-vali-path=build/simulation/Fold1/sim_exp_train_vali.csv --pred-ranking-dir=build/simulation/Fold1/sim_exp_train_vali_rankings.csv --nfeatures=136 --list-size=10`

### 4. Generate simulation clicks
1. Open and run through notebook `notebook/1_generate_clicks.ipynb`

### 5. Train bias estimation models
1. Generate input data: open and run through notebook `notebook/2_input_preprocess_bias_estimation.ipynb`
2. Train CPBM model: open and run through notebook `notebook/3.1_cpbm_train_and_test.ipynb`
3. Train Causal forests model: open and run through notebook `notebook/3.2_causal_forests_train_and_test.ipynb`
4. Train X-Learner model: open and run through notebook `notebook/3.3_x_learner_train_and_test.ipynb`

### 6. Parameter sweeping of LTR models
1. Sweep causal forests LTR:
    - run shell command: `invoke sweep-causal-forests --avg-clicks=5 --nq=159 --total-nqueries=15966 --fold=Fold2 --model-dir=build/simulation/Fold2/sweep --algorithm-name=cf_ltr --train-weights-feature-name=137 --eval-weights-feature-name=137 --batch-size=32 --reg-scale=0.01`
2. Sweep CPBM IPS-LTR:
    - run shell command: `invoke sweep-cpbm-ltr --avg-clicks=5 --nq=158 --total-nqueries=15966 --fold=Fold2 --model-dir=build/simulation/Fold2/sweep --algorithm-name=cpbm_ltr --train-weights-feature-name=137 --eval-weights-feature-name=137 --batch-size=32 --reg-scale=0.01`
3. Sweep X-Learner LTR:
    - run shell command: `invoke sweep-x-learner --avg-clicks=5 --nq=158 --total-nqueries=15966 --fold=Fold2 --model-dir=build/simulation/Fold2/sweep --algorithm-name=xlearner --train-weights-feature-name=137 --eval-weights-feature-name=137 --batch-size=32 --reg-scale=0.01`
4. Repeat or run multiple the above commands with different parameters, such as `--batch-size=64 --reg-scale=0.5 --avg-clicks=10 --nq=7983`. `notebook/invoke_task_generator.ipynb` can be used to generate such commands.


### 7. Train and Test LTR models
1. Get the best parameters from step 6.

2. Train and Test causal forests LTR:
    - run shell command: `invoke train-and-test-causal-forests-ltr --avg-clicks=5 --nq=159 --total-nqueries=15966 --fold=Fold2 --model-dir=build/simulation/Fold2/train_test/models --eval-result-dir=build/simulation/Fold2/train_test/test_results --algorithm-name=cf_binomial --test-data-path=build/simulation/Fold2/sim_test.txt --train-weights-feature-name=137 --epochs=6 --batch-size=32 --reg-scale=5.0`.

3. Train and Test CPBM LTR:
    - run shell command: `invoke train-and-test-cpbm-ltr --avg-clicks=5 --nq=159 --total-nqueries=15966 --fold=Fold2 --model-dir=build/simulation/Fold2/train_test/models --eval-result-dir=build/simulation/Fold2/train_test/test_results --algorithm-name=cpbm --test-data-path=build/simulation/Fold2/sim_test.txt --train-weights-feature-name=137 --epochs=16 --batch-size=32 --reg-scale=30.0`

4. Train and Test X-Learner LTR:
    - run shell command: `invoke train-and-test-x-learner-ltr --avg-clicks=5 --nq=159 --total-nqueries=15966 --fold=Fold2 --model-dir=build/simulation/Fold2/train_test/models --eval-result-dir=build/simulation/Fold2/train_test/test_results --algorithm-name=xlearner --test-data-path=build/simulation/Fold2/sim_test.txt --train-weights-feature-name=137 --epochs=6 --batch-size=32 --reg-scale=1.0`

Note: `notebook/invoke_task_generator.ipynb` can be used to get the best parameters and generate train and test tasks.
### 8. Result analysis
1. Open and run notebook: `notebook/4.1_results_analysis.ipynb`, `notebook/4.2_results_analysis.ipynb`
