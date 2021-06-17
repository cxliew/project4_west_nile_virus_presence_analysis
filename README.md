# # Project 4: West Nile Virus presence analysis
**By Clarence Thng, Thomas Lim, Liew Chin Xia**

### Problem Statement

Around the continental United States (US), the West Nile virus (WNV) is the leading mosquito-borne disease. Between 1999 to 2019, the yearly average number of WNV cases identified was around 2,500, with an average mortality rate of 5% [[1]](https://www.cdc.gov/westnile/statsmaps/cumMapsData.html).
As data scientists for the Chicago Department of Public Health (CDPH), we are seeking to create a classification model that predicts whether WNV will be present for a particular species of mosquito given the weather and location. This analysis is for the purpose of allocating CDPH resources more efficiently and effectively to combat and prevent the transmission of the WNV.
The team will explore the use of logistic regression, k-nearest neighbours, and random forest for the prediction of WNV in the mosquito populations.
As we are seeking to reduce the transmission of the disease, the model's success will be evaluated based on its F1 score, as we are looking to balance the number of false negatives and positives in order to make sure no resources are wasted while also ensuring that no clusters are missed out.

---
### Background

West Nile Virus(WNV) is a mosquito borne arbovirus that was first introduced in United States in 1999. It has became the most common virus transmitted by mosquitos (mainly *Culex* species) from infected bird to human in the U.S. with a mortality rate of 5% of reported cases[[1]](https://www.cdc.gov/westnile/statsmaps/cumMapsData.html)[[2]](https://us.biogents.com/house-mosquitoes/). Around 80% of the infected people does not display any symptoms while the 20% who are infected will develop West nile fever or lead to severe disease which are neuroinvasive disease [[3]](https://www.who.int/news-room/fact-sheets/detail/west-nile-virus). There are currently no vaccine available for humans and treatment is supportive for patients who developed neuroinvasive disease[[3]](https://www.who.int/news-room/fact-sheets/detail/west-nile-virus). Thus, the ways to reduce this spread is to raise awareness of prevention measures, implement mosquito surveillance and control programs in at-risk areas. [[4]](https://www.vdci.net/vector-borne-diseases/west-nile-virus-education-and-mosquito-management-to-protect-public-health/)

In Chicago, the first human case with wnv virus was reported in 2002 with 22 fatalities [[5]](https://www.chicago.gov/dam/city/depts/cdph/comm_dis/general/Communicable_Disease/CD_CDInfo_Jun07_WNV.pdf). As a preventive measure of this WNV outbreak, Chicago Department for Public Health (CDPH) has set up a surveillance and control program that includes annual spraying in 2004.[[6]](https://www.govtech.com/analytics/chicago-turns-to-predictive-analytics-to-map-west-nile-threat.html). In order to spray areas having WNV-carrying mosquitoes, a predictive model is required. As data scientists for CDPH, we are seeking to create a classification model that predicts whether WNV will be present for a particular species of mosquito given the weather and location. The team will explore the use of logistic regression, k-nearest neighbours, and random forest for the prediction of WNV in the mosquito populations. As we are seeking to reduce the transmission of the disease and allocating CDPH resources more efficiently and effectively, the model's success will be evaluated based on its F1 score. This will allow us to balance the number of false negatives and positives in order to make sure no resources are wasted while also ensuring that no clusters are missed out.


---
### Data Dictionary

The dataset contains the weather, location, testing and spraying in the City of Chicago.The data source below are obtained from [kaggle](https://www.kaggle.com/c/predict-west-nile-virus/data).

The datasets obtained are as followed:-

* train_df (2007, 2009, 2011, 2013)
* spray_df (2011 to 2013)
* weather_df (2007 to 2014)
* test_df (2008, 2010, 2012, 2014)

The data dictionary for the data analysis and modeling are as followed:-

|Feature|Type|Dataset|Description|Remarks|
|:---|:---|:---|:---|:---|
|**date**|*object*|train_df<br>test_df|The date that the West Nile Virus test is performed||
|**species**|*object*|train_df<br>test_df|The species of mosquitos<br>0 : UNSPECIFIED CULEX<br>1 : CULEX ERRATICUS<br>2 : CULEX PIPIENS<br>3 : CULEX PIPIENS/RESTUANS<br>4 : CULEX RESTUANS<br>5 : CULEX SALINARIUS<br>6 : CULEX TARSALIS<br>7 : CULEX TERRITANS|Serialised|
|**latitude**|*float64*|train_df<br>test_df|Latitude returned from GeoCoder||
|**longitude**|*float64*|train_df<br>test_df|Longitude returned from GeoCoder||
|**wnvpresent**|*int64*|train_df|Whether West Nile Virus was present in these mosquitos.<br>1 means West Nile Virus is present, and 0 means not present||
|**tmax**|*float64*|train_df<br>test_df|Maximum temperature (&deg;F)||
|**tmin**|*float64*|train_df<br>test_df|Minimum temperature (&deg;F)||
|**tavg**|*float64*|train_df<br>test_df|Average temperature (&deg;F)||
|**dewpoint**|*float64*|train_df<br>test_df|Average dew point (&deg;F)||
|**wetbulb**|*float64*|train_df<br>test_df|Average wet bulb (&deg;F)||
|**heat**|*float64*|train_df<br>test_df|Heating degree days (Base 65&deg;F)||
|**cool**|*float64*|train_df<br>test_df|Cooling degree days (Base 65&deg;F)||
|**preciptotal**|*float64*|train_df<br>test_df|Total precipitation (Inches and Hundredths)||
|**stnpressure**|*float64*|train_df<br>test_df|Average station pressure (Inches of Hg)||
|**sealevel**|*float64*|train_df<br>test_df|Average sea level pressure (Inches of Hg)||
|**resultspeed**|*float64*|train_df<br>test_df|Resultant wind speed (miles per hour)||
|**resultdir**|*float64*|train_df<br>test_df|Resultant wind direction (tens of whole degrees)||
|**avgspeed**|*float64*|train_df<br>test_df|Average wind speed (miles per hour)||
|**saturation_vapour_pressure**|*float64*|train_df<br>test_df|Saturation of vapour pressure (mbar)||
|**actual_vapour_pressure**|*float64*|train_df<br>test_df|Actual vapour pressure (mbar)||
|**relative_humidity**|*float64*|train_df<br>test_df|Relative humidity (percentage expressed as a decimal)||
|**codesum**|*object*|train_df<br>test_df|Weather phenomena for significant weather types|Dummified|
|**trap**|*object*|train_df<br>test_df|Id of the trap|Dummified|

---
### Data Cleaning

The datasets are cleaned as followed:-

* The column names and values have been changed to lower case.
* The selected columns have been removed as majority has missing values or these columns will not be used in our analysis.
* The missing values handled by calculating and imputing, while some values are removed.
* Remove the outliers for trap locations.
* The change in data-types for numerical values.

---
### Exploratory Data Analysis

**Feature engineering**

We have performed feature engineering as listed below:-
1. Create new features such as relative humidity, saturation vapour pressure and actual vapour pressure.
2. Average the weather data of two stations by dates
3. Dummify the trap, weather phenomena and species of mosquitos
4. Oversample the data to combat imbalance classes.


**Analysis**
1. We found a high correlation between the number of mosquitos and weather features such as precipitation, relative humidity, temperature, as well as some weather phenomena - for example rain, drizzle, and thunderstorms

2. The mosquito populations peaked between July and August each year. This coincided with the time of the year when it started to get warmer, and there was more rain and humidity

3. Two species of mosquitos in particular had a much higher population than the others - Culex Pipiens and Culex Restuans.

4. There was a higher mosquito population in certain locations, but these locations were not where spraying efforts were previously done.

These will likely be the key features for our model in predicting the probability of WNV presence for given locations, time periods and species of mosquitos.

----
### Data Preprocessing and Modeling & Analysis Summary

As this project aims to develop a model that are able to predict whether or not west nile virus is present given a location, time and species, it is a categorical classification.

Thus, the models that we will be using are:-
* Logistic regression
* k-nearest neighbours classifier
* Random forest classifier

Of the 3 models, we picked Random Forest Tree as our final model. This is becuase not only does it have the highest AUC Score of 99%, it also has the highest F1 Score which is what we based our model performance on.

We think Random Forest Tree(RTF) worked best in this situation because there is a requirement of multiple factors occurring in order that may contribute to the population of the mosquitoes, thereby indirectly increasing the probability of WNV presence among them. However, should any one of the factor sequences or groups be broken, it may produce the opposite results instead. RFT is basicly a decision tree, and at the end of each decision tree is the resulting probability of that event happening. While progressing through the depth of the decision tree, a probability of a result happening can swing from one end of the scale to the other end drasticly.  

---

### Cost Benefit Analysis
Chicago is a very large area and at the cost of $500 per spray, it would not make sense to conduct sprays over the entire city. From the data given, there were 4000 sprays done in July and Sep and close to double the amount in Aug as that's when the population is the highest. that would cost an estimated total of $8 million across the 3 months.

As we have approx 130 traps around the city, if we can accurately identify locations where the virus is likely to be, and conduct sprays around the area and 9 surrounding locations once a week, even if the model predicts 50% of the trap locations to likely have the virus present, that would still only be approx 8450 sprays over 3 months, or $4.225 million, which is effectively saving slightly less than 50% of the cost.   

$500 * 130 traps * 0.5 (50% prediction rate) * 10 locations per trap * 13 weeks (3 months) = $4,225,000

---

### Conclusion

In our EDA, we have shown how the number of mosquitoes has a correlation to various weather condition as well as how the impact of spraying pesticide is able to effectively reduce the number of mosquitoes in an area. We engineered additional features which could further help us with our modeling process.

We performed 3 different types of modelling process and compared them with each other before finally determining that Random Forest Tree was the best model in this case. We also shown that our model is able to perform accurately to a high degree, performing better than our baseline of 95% accuracy and a high F1 score of 96%.  

This means that our model is able to correctly predict if WNV will be found to be present given a location and time based on the weather conditions roughly 96% of the time. The model also manages to keep the false negatives and positives at a minimum, so that population outbreaks are not missed, and efforts are not wasted in spraying locations with less probability of an outbreak.

Should our model be implemented as well, our cost benefit analysis suggest that nearly 50% of savings can be achieved.

---

### Recommendations
Based off our EDA spraying in an area does help reduce the number of mosquitoes in the area. Thus, our recommendation is to continue to engage regular spraying throughout the period with more frequency around July to August as it has been noted that this is the period where the population of mosquitoes is the highest.

Use our model to predict which locations are more likely to have the virus present and spraying can be concentrated within those areas to maximise the effectiveness of the sprays and minimize the cost.

---
### Future Improvement

Some future work that could be done is do a deeper look into the features selected for modeling. One of them could be removing highly correlated features. Another would to be to gather more data on the effects of spray data. Having more data collected will help to combat any overfitting issues.

---
### Kaggle Submission

Along our main problem statement, we were also aiming to submit our predictions to Kaggle to see how well our model does.The Kaggle submission involves predicting if WNV will be present based on a "when" and a "where", which is inline with our problem statement.  

However, our goal was not to optimize the model for Kaggle, but instead to optimize for the minimization of false positive and negatives. For each model, we made submissions to Kaggle, and it was interesting to see how the best performing model in terms of F1 score was not the best performing model for Kaggle.  

This is likely due to some overfitting occurring in the models. Given that the test dataset has a new 'unspecified species' and traps which differ from the training data set, the differences in location and species may have affected most of the models. It also showed us how optimizing models for Kaggle may not always optimize well for the problem statements we might have on hand.  

Overall, the Kaggle submissions were a learning experience for the group.

---
### References

[1] "West Nile virus disease cases reported to CDC by state of residence, 1999-2019," *Centers for Disease Control and Prevention*, November 24, 2020. [Online]. Available: [https://www.cdc.gov/westnile/statsmaps/cumMapsData.html](https://www.cdc.gov/westnile/statsmaps/cumMapsData.html) [Accessed: May. 6, 2021].

[2] "House Mosquitoes," *Biogents USA*. [Online]. Available: [https://us.biogents.com/house-mosquitoes/](https://us.biogents.com/house-mosquitoes/) [Accessed: May. 6, 2021].

[3] "West Nile Virus," *World Health Organization*, October 03, 2017. [Online]. Available: [https://www.who.int/news-room/fact-sheets/detail/west-nile-virus](https://www.who.int/news-room/fact-sheets/detail/west-nile-virus) [Accessed: May 6, 2021].

[4] "What is West Nile Virus and How Does it Spread?," *VDCI A Rentokil Company*, 2021. [Online]. Available: [https://www.vdci.net/vector-borne-diseases/west-nile-virus-education-and-mosquito-management-to-protect-public-health/](https://www.vdci.net/vector-borne-diseases/west-nile-virus-education-and-mosquito-management-to-protect-public-health/) [Accessed: May 6, 2021].

[5] "West Nile Virus," *Chicago Department of Public Health Communication Disease Information*, June, 2007. [Online]. Available: [https://www.chicago.gov/dam/city/depts/cdph/comm_dis/general/Communicable_Disease/CD_CDInfo_Jun07_WNV.pdf](https://www.chicago.gov/dam/city/depts/cdph/comm_dis/general/Communicable_Disease/CD_CDInfo_Jun07_WNV.pdf) [Accessed: May 6, 2021].

[6] S. Thornton, "Chicago Turns to Predictive Analytics to Map West Nile Threat," *Government Technology*, October 25, 2017. [Online].[https://www.govtech.com/analytics/chicago-turns-to-predictive-analytics-to-map-west-nile-threat.html](https://www.govtech.com/analytics/chicago-turns-to-predictive-analytics-to-map-west-nile-threat.html) [Accessed: May 6, 2021].

---
