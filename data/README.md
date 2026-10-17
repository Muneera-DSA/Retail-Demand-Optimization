# Data

| Part | Folder | Files | In this repository? |
|---|---|---|---|
| A: TCS iON product × outlet demand | `data/tcs/` | `train_data.csv`, `product_prices.csv`, `date_to_week_id_map.csv` | **No**: course material, see below |
| B: Favorita | `data/favorita/` | `train.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `transactions.csv`, `test.csv` | **No**: Kaggle competition data, download it yourself |
| C: hourly demand | `data/hourly/` | `hourly_demand.csv` | Yes (300 KB) |

## Part B: Favorita (Corporación Favorita, Ecuador)

Source: Kaggle, *Store Sales – Time Series Forecasting*
(<https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data>). The data may only be used under
the competition rules, so it is not redistributed here. To reproduce Part B:

1. Sign in to Kaggle, open the link above, click **Join Competition** and accept the rules (no submission is needed).
2. **Download All**, unzip, and put the six CSV files in `data/favorita/`.

| File | Rows | Content |
|---|---|---|
| `train.csv` | 3,000,888 | daily unit sales and items on promotion, per store (54) × product family (33), 2013-01-01 to 2017-08-15 |
| `test.csv` | 28,512 | the next 16 days, promotions only (no sales): used only for planned promotions |
| `stores.csv` | 54 | city, state, store type (A-E), cluster |
| `oil.csv` | 1,218 | daily WTI oil price (Ecuador's economy is oil-dependent); weekdays only, 43 missing |
| `holidays_events.csv` | 350 | national, regional and local holidays and events, with transfers, bridge days and compensating work days |
| `transactions.csv` | 83,488 | daily transactions per store |

Notes found by the data checks (`results/favorita/quality.json`): 25 December is absent every year (stores closed);
8 stores opened during the period; promotions are recorded only from April 2014; the 16 April 2016 earthquake
(Manabí) produced relief-buying spikes in the following weeks.

## Part C: hourly demand

`hourly_demand.csv`: columns `date`, `hour` (0-23), `demand` (integer count), 2018-08-18 to 2021-02-28,
18,247 rows. **The origin and unit of this series are not documented**; it is analysed as a generic hourly demand
signal for capacity planning. Hours with no record are absent from the file (there are no zeros); see the data
checks for how they are treated.

## Part A: TCS iON retail sales-volume project

The files come from the TCS iON industry project "Forecasting demand for products across retail outlets" and were
used in my original (v1) notebook. They are **not redistributed here** (course material). To reproduce Part A, put
these files in `data/tcs/`:

| File | Rows | Content |
|---|---|---|
| `train_data.csv` | 395,000 | `date`, `product_identifier`, `department_identifier`, `category_of_product`, `outlet`, `state`, `sales`: daily unit sales, 50 products × 10 outlets, 2012-01-01 to 2014-02-28 |
| `product_prices.csv` | weekly | `outlet`, `product_identifier`, `week_id`, `sell_price` |
| `date_to_week_id_map.csv` | 1 per day | `date`, `week_id` |

`test_data.csv` and `sample_submission.csv` (the course's March 2014 test period, no sales) are not used: they have no
actuals to evaluate against. The data checks (`results/tcs/quality.json`) found a complete, valid grid with 62% zero
days, prices that change in about 1% of weeks, and 72 series with more than 180 days in a row without a sale.
