# psmatching2
Enhanced-PSMatching for Python

Propensity score matching(PSmatching) is a commonly used statistical method in the fields of medicine and public health. It aims to control for confounding biases and improve the effectiveness of analyzing the relationship between exposure factors and outcome measures. Currently, this method is primarily applied in the R language, while the matching implementation in the third-party library "psmatching" in Python is relatively limited, with restricted application scenarios. Therefore, we have replicated the psmatching library and referenced the functionality and methods available in R. We have successfully implemented propensity score matching and added multiple weighting methods to enhance its capabilities.

##Get Started
```sh
# Clone this repo
git clone https://github.com/YanKong0408/psmatching2.git
cd psmatching2

# Install Pytorch and torchvision
# We test our models under 'python=3.7.3,pytorch=1.9.0,cuda=11.1'. Other versions might be available as well.
conda install -c pytorch pytorch torchvision

# Install other needed packages
pip install -r requirements.txt
```
