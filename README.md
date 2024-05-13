# psmatching2
Enhanced-PSMatching for Python

Propensity score matching(PSmatching) is a commonly used statistical method in the fields of medicine and public health. It aims to control for confounding biases and improve the effectiveness of analyzing the relationship between exposure factors and outcome measures. Currently, this method is primarily applied in the R language, while the matching implementation in the third-party library "psmatching" in Python is relatively limited, with restricted application scenarios. Therefore, we have replicated the psmatching library and referenced the functionality and methods available in R. We have successfully implemented propensity score matching and added multiple weighting methods to enhance its capabilities.

The following functionality is included in the package:
* Calculation of propensity scores based on a specified model.
* Matching of k controls to each treatment case with four different methods.
* Use a caliper to control each treatment case.
* Matching with or without replacement.
* Performing weight processing on matched data.
* Calculate the p value on the basis of the statistic.
* Evaluation of the matching process using statistical methods.
## Get Started
```sh
# Clone this repo
git clone https://github.com/YanKong0408/psmatching2.git
cd psmatching2

# Install other needed packages
pip install -r requirements.txt
```

## Main Functions
```sh
#Instantiate PSMatch object
m=PSMatch(path,control, covariants, dependents=None)
# Calculation of propensity scores based on a specified model
m.calculate_proprnsity_scores()
# Matching of k controls to each treatment case with four different methods.
m.match(caliper=None, replace=False, k=1)
# Performs weight process.
m.weighted_process(self,method=None)
# Calculate the p value on the basis of the statistic.
m.evaluate_p_value(self, df,if_show=True)
# Evaluate the mean, variance, SMD of coveriates and the results of dependent variables
m.evaluate_dependent(df)
# show the matching effect of each method
m.plot_matching_efficiency(if_weighed=True)
```

## Example
Below is an example showcasing the application of our method to evaluate the effectiveness of right heart catheterization procedure.
Code is in '[test_file.py](https://github.com/YanKong0408/psmatching2/blob/main/test_file.py)'.
Data can be download at [here](https://hbiostat.org/data/).
Result shows as below:
![Intro](./image/smd.png)
![Intro](./image/failed_pvalue.png)
