## Introduction
This is for Ecesis Investments **2026 Summer Power Systems Modeling Internship** recuriting. 
- You can use either **R** or **Python** (Recommended) for coding.
- Requirements:
  - create your own github repository for result delivery
  - a summary report for each assignment is required.
  - for python, create jupyter notebook file
- **Due:** early delivey is welcomed.
- Hints:
  - make your code neat and self explanatory
  - in summary report, provide your insights and conclusions

## Assignment 1: Constraint Mapping Across Data Sources
#### Objective: map constraints across 3 different sources, identifying which entries correspond to the same underlying physical constraint
In power markets, the same physical constraint rarely speaks with a single name. Different vendors and systems describe constraints using their own naming conventions, formats, and levels of granularity. For a trading desk, this creates a critical problem: if you cannot reliably match constraints across sources, you cannot form a coherent view of congestion risk.
#### Data Provided:
In this assignment, you are given three constraint lists from different sources:
  - PJMISO market data
  - Dayzer
  - Panorama
Each source represents constraints using slightly different structures. In particular, constraints are defined by **(facility, contingency)** pairs, but naming conventions and formatting may differ across datasets. Think of this as building a translation layer between three dialects of the same grid language.
#### Output Requirements:
  - csv file with matched results (market_constraint, dayzer_constraint, pano_constraint)
  - jupyter notebook showing process and comments
  - summary report with insights and conclusions
#### Hints:
Strong solutions typically combine:
  - string matching and normalization
  - domain-informed heuristics
  - careful handling of ambiguous or partial matches

## Assignment 2: Bus Mapping via Network Structure
#### Objective: map each Dayzer bus to its corresponding Panorama bus
A power grid is not just a collection of nodes, but a living network of relationships. While individual buses may look similar across datasets, their positions within the network sometimes set them apart. Two buses with slightly different names may in fact be the same node, if their electrical “neighborhood” aligns.
#### Data Provided:
In this assignment, you are given:
  - **Bus** lists from Dayzer and Panorama (including metadata such as name, kV level, latitude, and longitude)
  - **Branch** lists from both sources, describing how buses are connected
You can think of this as aligning two different maps of the same city, where street names may differ, but the structure of intersections and roads remains consistent.
#### Output Requirements:
  - csv file with matched results (dayzer_bus, pano_bus, and extra columns you find useful)
  - jupyter notebook showing process and comments
  - summary report with insights and conclusions
#### Hints:
This problem is intentionally open-ended. There is no single "correct" mapping method. Unlike the first assignment, this task benefits from thinking beyond individual records.
Useful signals may include:
  - name similarity
  - voltage levels
  - geographic proximity
  - network topology (how buses are connected)
We are less interested in perfect accuracy and more interested in how you approach this problem.