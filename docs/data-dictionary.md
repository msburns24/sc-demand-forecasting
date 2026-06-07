# Data Dictionary

## Calendar

- **Filename:**  `calendar.csv`
- **Rows:**      1,969
- **Columns:**   14

| Column Name  | Type    | Comment                                             |
| :----------- | :------ | :-------------------------------------------------- |
| date         | date    | Format: `YYYY-MM-DD`                                |
| wm_yr_wk     | integer | Week ID (`11101` - `11621`)                         |
| weekday      | string  | Weekday name (Saturday, ..., Friday)                |
| wday         | integer | Weekday ID (`1` = Saturday, ..., `7` = Friday)      |
| month        | integer | Month number (`1`, `2`, ..., `12`)                  |
| year         | integer | `2011` - `2016`                                     |
| d            | string  | Date index (`d_1`, `d_2`, ..., `d_1969`)            |
| event_name_1 | string  | E.g., `SuperBowl`, `ValentinesDay`                  |
| event_type_1 | string  | Cultural, National, Religious, or Sporting          |
| event_name_2 | string  | If date has 2nd event                               |
| event_type_2 | string  |                                                     |
| snap_CA      | bool    | If CA stores allow SNAP purchases on the given date |
| snap_TX      | bool    | If TX stores allow SNAP purchases on the given date |
| snap_WI      | bool    | If WI stores allow SNAP purchases on the given date |

## Sales

- **Filename:** `sales_train_evaluation.csv`
- **Rows:**     30,490
- **Columns:**  1,941

| Column Name | Type    | Comment                                           |
| :---------- | :------ | :------------------------------------------------ |
| id          | string  | Format: `{item_id}_{store_id}_evaluation`         |
| item_id     | string  | E.g., `HOBBIES_1_001`                             |
| dept_id     | string  | E.g., `HOBBIES_1`, `FOODS_3`                      |
| cat_id      | string  | `HOBBIES`, `HOUSEHOLD`, or `FOODS`                |
| store_id    | string  | `CA_1`, `CA_2`, ..., `WI_3`                       |
| state_id    | string  | `CA`, `TX`, or `WI`                               |
| d_1         | integer | Number of units sold on day `d_1` (2011-01-29)    |
| d_2         | integer | Number of units sold on day `d_2` (2011-01-30)    |
| ...         | ...     | ...                                               |
| d_1941      | integer | Number of units sold on day `d_1941` (2016-05-22) |

## Prices

- **Filename:**  `sell_prices.csv`
- **Rows:**      6,841,121
- **Columns:**   4
- **Known Quirks:**
  - Only contains data when the item was sold

| Column Name | Type    | Comment                             |
| :---------- | :------ | :---------------------------------- |
| store_id    | string  | `CA_1`, `CA_2`, ..., `WI_3`         |
| item_id     | string  | E.g., `HOBBIES_1_001`               |
| wm_yr_wk    | integer | Week ID (`11101` - `11621`)         |
| sell_price  | float   | Item price for the given week/store |
