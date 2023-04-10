# A Look into King County Home Improvements

![King County picture of shore line with houses](https://github.com/cschneck7/phase-2-project/blob/main/images/hero-lake-washington-xlg.jpg)

#### Links

[Report Notebook](Report.ipynb)<br/>
[EDA and Modelling Notebook](EDA_and_Modelling.ipynb)<br/>
[Presentation](Presentation.pdf)

## Overview

This project analyzes King County housing sales data in order to help lead a wholesale real estate investor make educated decisions on which home improvements best improve sales prices in the area. King County is the most populous county in Washington State, and 13th in the country.

## Business Understanding

A wholesale real estate investor wants to get a better idea of which improvements relate to the the biggest increase in sales price. By finding out this information they can better access which projects are more worthwile and lead to the largest profit margin. Through a linear regression analysis we can figure out which features have the largest affect on sales price by looking at our models coefficients.

The area of interest is King County, Washington. Which is the most populous county of Washington state, and ranked 13th in the country. In order to complete the objective a dataset containing sales data for King County spanning the years 2014 to 2015 was analyzed.

## Data Understanding

Our dataset includes information on houses sold in the timeframe spanning the years 2014 through 2015 and 70 zipcodes of King County. There are a total of 21,597 entries with 21 columns worth of information. For this analysis we will cut down this data to features we believe will be helpful for the client. These include features that can be improved after the purchase of a home. Features dealing with location, view or neighbooring properties are things that in most cases are impossible to change so will be ommited.

## Modeling

The model choes was a linear regression model. We came to this decision based on the goal of the project. We wanted to be capable of inferring information from each feature, thus needing a simpler model that could be understood.

We started modeling with seven features but ended with only three features.

- `sqft_living` - Square footage of living space
- `condition` - Overall maintanence condition of the house
  - 1 = Poor
  - 2 = Fair
  - 3 = Average
  - 4 = Good
  - 5 = Very Good
- `grade` - Overall construction and design grade of the house
  - 3 = Poor
  - 4 = Low
  - 5 = Fair
  - 6 = Low Average
  - 7 = Average
  - 8 = Good
  - 9 = Better
  - 10 = Very Good
  - 11 = Excellent
  - 12 = Luxury
  - 13 = Mansion

The top two features making an impact on the change in house price were `sqft_living` and `grade`. Respectively increasing the house price by $104.63 and $96,872.70 per unit of improvement. The models R<sup>2</sup> value was .521 thus explaining around 52.1% of our target variables variance. The Root Mean Squared Percent Error was pretty high at about 41%.

The model struggled with passing all the assumptions, performing poorly on the normality and homoscedasticity checks. Though the model didn't perform well on these tests it is the best at this time. Any improvements made would complicate the model to a point that it is hard to understand.

## Conclusion

This models goal was to find the features that had the largest affect on changing the price of a home. The approach was to only include variables that were deemed changeable, then build our model using these features. The final model has a Root Mean Squared Percent Error (RMSPE) of around 41% and an R<sup>2</sup> score of .521 therefore explaining around 52% of our target variables variance. The top two features were determined to be `sqft_living` and `grade`. A change of one `sqft_living` unit equates to a \$104.63 increase in price. While a single improvement in `grade` equates to a change of \$96,872 in price. Therefore it is suggested to either increase the square footage of the living space, or to remodel the living space to increase the grade of the home. Though before taking any action an analysis of material and labor costs should be taken into consideration. Since this model had a high RMSPE and didn't pass all the assumptions, it is hard to recommend its use with full confidence.

## Future Improvements

This model didn't perform too well on all of the assumptions for a linear regression model. The accuracy of only 41% is also too high to make a confident prediction. Therefore some improvements may be made if the following is performed.

#### Include more Features

The initial approach to this model was to strip it of all the variables that were deemed unchangeable. This included location and land descriptions. Though these variables couldn't be picked for home improvements, they could have helped narrow the model coefficients down and improved accuracy.

#### Make multiple models for different feature categories

There are possibly some features that play a large part in the difference in price, for example `waterfront` or `zipcode`. It may be beneficial to split the data into seperate models by features like these to determine if different features play a larger part in these categories.

#### Further limit the range of the model

Looking at the homoscedastic check, the residuals on the tail with houses of higher sale value are larger than those of lower values causing it to be heteroscedastic. Thus we could possibly limit the range of the model even further and make two different models for different ranges.

#### Try the box-cox method

The target variable `price` and feature `sqft_living` appear to have logarithmic curves. While this analysis attempted to transform the data using a log function it didn't pass the assumption test for homoscedasticity. The plot of residuals over the range showed a variance that was trending upwards. Upon research this could possibly be fixed using the box-cox method. The downside being it is not recommended if the goal is to infer information from the the individual feature coefficients since the transformation is exponential.

## Repository Structure

├── data<br/>
├── images<br/>
├── EDA_andModelling.ipynb<br/>
├── Report.ipynb<br/>
├── Presentation.pdf<br/>
├── **init**.py<br/>
├── data_preparation_functions.py<br/>
├── figure_functions.py<br/>
└── README.md<br/>
