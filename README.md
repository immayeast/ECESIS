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
#### Estimated Time: 5 to 8 hours.
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



## Assignment 2: Bus-level Load Prediction Pipeline

#### Estimated Time: 1 week.
#### Objective: Build an End to End Bus-level Load Prediction Pipeline for Next Day and Next Month with Modern AI.

> [!IMPORTANT]
> You are encouraged to fully use modern AI coding platforms such as **Codex**, **Claude Code**, ChatGPT, Cursor, or similar tools to complete this project.
>
> AI tools may be used for coding, debugging, data exploration, feature engineering, model comparison, documentation, and report drafting. However, you are responsible for the final submission and should be able to explain, validate, and defend all code, forecasts, evaluation results, and design decisions.
>
> The company will provide access to a **Codex account** if you do not have access to any of the modern AI platform. Let us know if you need one.
>
> Data will be provied via Google Drive, please share your email with us so that we can give you access to the data.


Electricity demand changes over time by location, hour, weekday, season, and broader system conditions. In this project, each bus represents a location where load is measured, and each zone represents a group of related buses.

Your goal is to build an end-to-end forecasting pipeline that predicts hourly load for every bus. The pipeline should support two use cases: a short-term next-day forecast and a longer-term next-month forecast. This is a grouped time-series forecasting problem, so no prior power-market knowledge is required.

Each **bus** is one measurement location. Each **zone** is a group of buses.

`pd` means load(demand) in MW.

`pg` means generation in MW.

`HE` means hour-ending. Usually `HE1` means `00:00:00 to 00:59:59`, `HE24` means `23:00:00 to 23:59:59`.

Build a forecasting pipeline for two tasks:

1. **Next-day forecast**
   - Given data up to a forecast date, predict the next day.
   - Example: on `2025-05-11 HE7`, predict hourly load for `2025-05-12 HE1` through `2025-05-12 HE24`.

2. **Next-month forecast**
   - Given data up to the first day of a month, predict the full next month.
   - Example: on `2025-05-01`, predict hourly load for all of June 2025.

The goal is to build a clean, reproducible forecasting workflow. In the end of the project, you should report the workflow performance based on the instructions provided below.





#### Data Provided:

You are given two types of files.

##### Hourly Bus Load Data

| id | bus_unique_id | bus_type | base_kv | zone_name | pd | pg | date | he |
|---:|---|---|---:|---|---:|---:|---|---:|
| 1 | TNTEPCOT_69KV_1 | LOAD | 69.0 | COAS |  |  | 2022-01-01 | 1 |
| 2 | TNMRTHPT_69KV_1 | LOAD | 69.0 | COAS |  |  | 2022-01-01 | 1 |
| 3 | RICEBI_138KV_1 | LOAD | 138.0 | COAS |  |  | 2022-01-01 | 1 |
| 4 | RICEBI_69KV_1 | LOAD | 69.0 | COAS |  |  | 2022-01-01 | 1 |
| 5 | VANHUM_69KV_1 | LOAD | 69.0 | COAS | 0.129 |  | 2022-01-01 | 1 |
...

##### Hourly Zone Load Data

| zone_name | pd | pg | load_bus_count | gen_bus_count | date | he |
|---|---:|---:|---:|---:|---|---:|
| COAS | 11518.576 | 9877.01 | 810 | 232 | 2022-01-01 | 1 |
| EAST | 1343.929 | 5914.732 | 241 | 36 | 2022-01-01 | 1 |
| FWES | 3837.229 | 1171.28 | 531 | 125 | 2022-01-01 | 1 |
| ISOLATED | 0.0 | 0.0 | 0 | 0 | 2022-01-01 | 1 |
| NCEN | 9508.056 | 5675.701 | 1191 | 109 | 2022-01-01 | 1 |
...


You may create additional features from the provided data, such as:

- hour of day
- day of week
- month
- weekend or holiday flags
- lagged load values
- rolling averages
- historical bus share within each zone

#### 3. Train/Test Setup

The two forecasting tasks should be evaluated separately. In both cases, the test targets should cover **full calendar year of 2025**.

##### Task 1: Next-Day Forecast

Use historical data before the forecast date to predict the next day demand for each bus per he.

```text
Training period: 2022-01-01 through 2024-12-31
Test target period: 2025-01-01 through 2025-12-31
```

For each day in 2025:

```text
forecast_created_at = previous day
target = next 24 hours
```

Example:

```text
forecast_created_at: 2025-05-11
target period: 2025-05-12 HE1 through 2025-05-12 HE24
```

The model may use load history available up to `forecast_created_at`, but must not use actual load from the target day.

##### Task 2: Next-Month Forecast

Use historical data before the forecast date to predict the full following month per target date and he.

```text
Training period: 2022-01-01 through 2024-12-31
Test target period: 2025-01-01 through 2025-12-31
```

For each month in 2025:

```text
forecast_created_at = first day of the previous month
target = full next month
```

Examples:

```text
forecast_created_at: 2024-12-01
target period: 2025-01-01 through 2025-01-31

forecast_created_at: 2025-05-01
target period: 2025-06-01 through 2025-06-30
```

The forecast should include every hour of the target month for every bus.

Important: for each forecast, only data available on or before `forecast_created_at` may be used.

#### Output Requirements:

- runnable code

- forecast output files

   Use this required schema:

   | model_name | forecast_created_at | target_date | he | bus_id | zone_id | predict_pd |
   |---|---|---|---:|---|---|---:|
   | my_model | 2025-05-11 00:01:00 | 2025-05-12 | 1 | US_001 | ZONE_A | 13.2 |
   | my_model | 2025-05-11 00:01:00 | 2025-05-12 | 1 | US_002 | ZONE_A | 15|
   ...
         

- a short report
   
      Briefly explain:
      - what model you used
      - what baseline you compared against
      - what features you created
      - how you avoided using future data
      - where the model works well
      - where it performs poorly
      - what you would improve with more time

      Some metrics to use:
      MAE = mean(abs(actual - forecast))
      RMSE = sqrt(mean((actual - forecast)^2))
      WMAPE = sum(abs(actual - forecast)) / sum(actual)

      Evaluate separately for:
      - next-day forecast
      - next-month forecast
      - overall bus-level accuracy
      - zone-level aggregated accuracy

- AI Usage Logs
   - explain how you used AI tools to finish the task.
   - submit chat history.


#### Hints / Suggested Approach:

Start simple.
A good solution can follow this path:

1. **Build a baseline**
   - Same hour from previous week
   - Same hour from previous year
   - Historical average by bus, hour, and weekday

2. **Build a machine learning model**
   - Use features such as hour of day, day of week, month, weekend flag, previous load values, rolling averages, `bus_id`, and `zone_id`.

3. **Try two forecasting strategies**
   - **Direct bus forecast:** predict each bus directly.
   - **Zone forecast + bus share:** predict total zone load, then distribute it to buses based on historical bus shares.

4. **Compare results**
   - Which method works better for next-day forecasts?
   - Which method works better for next-month forecasts?
   - Why?

Important: avoid data leakage. A forecast made on a given date should only use information available on or before that date.



## (Optional) Assignment 3: Bus Mapping via Network Structure

It is welcomed to only provide thoughts on how to tackle this question if you have no capacity to do it.
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
