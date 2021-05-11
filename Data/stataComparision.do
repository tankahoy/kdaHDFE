/*
Author : Samuel Baker
Version: 0.1.0
Date   : 11/05/2021
Paper  : ExampleData run via reghdfe
Purpose: This Script is design as a point of comparision from reghdfe and the
		 python model of FixedEffectModels forked on kdaHDFE
Notes  : Has the following Dependencies, that can install by coping these lines
		 and placing them in the command bar.
		 
		 ssc install reghdfe
		 
*/
clear all
set more off

import delimited "C:\Users\Samuel\PycharmProjects\kdaHDFE\Data\ExampleData.csv"

* Just some testing parameters for comparision
reghdfe rs012 bmi gender, absorb(pob)
reghdfe rs012 bmi gender, noabsorb cluster(pob)
reghdfe rs012 , absorb(pob)
reghdfe rs012, noabsorb cluster(pob)
reghdfe rs012 bmi gender, absorb(pob yob)
reghdfe rs012 bmi gender, absorb(pob yob i.pob##i.yob)


