# Inventory Policy

## Purpose

This document defines the inventory policy for SKU-store combinations in the
supply chain demand forecasting system. It explains the ABC-XYZ segmentation
framework, maps each segment to a stocking policy (MTS vs. MTO), and assigns
target service levels that drive downstream safety stock calculations.

<br />

## Background

### ABC Classification - Pareto by Volume

An **ABC Classification** categorizes inventory into 3 priority groups based on
their **volume**:

| **Class** | **Criteria**         | **Description**                 |
| :-------: | :------------------- | :------------------------------ |
| A         | Top 70% of volume    | Best-selling products           |
| B         | Middle 20% of volume | Middle-ground                   |
| C         | Bottom 10% of volume | Low-value / high-quantity items |

Some notes:

1. **The ABC criteria isn't always 70/20/10.** The pareto principle is
   traditionally 80/20, but given we have 3 groups instead of 2, I'm selecting
   70/20/10. However, some supply chain organizations choose other splits, like
   80/15/5.
2. **Here, volume is defined by sales amount (in USD).** Unlike the criteria
   split points, this rarely differs at different companies, since sale price
   can vary widely between items, making sales units unfit for the measurement.
   For example, if a store sold 500,000 packs of gum and 100,000 units of
   Apple AirPods, we wouldn't want the gum to rank higher than the AirPods,
   since AirPods cost roughly 100x as much.

### XYZ Classification - Pareto by Variability

An **XYZ Classification** categories inventory into 3 priorities based on
their variability, or more specifically, their **coefficient of variation (CV)**:

$$CV = \frac{\sigma_{\text{weekly demand}}}{\mu_{\text{weekly demand}}}$$

Demand variability directly drives safety stock requirements. A SKU with
predictable demand needs only a small buffer to maintain a high service level. An
erratic SKU requires disproportionately large safety stock to achieve the same
service level — often making it more economical to manufacture or order only when
a customer actually places an order. By quantifying variability through CV, the
XYZ classification identifies which items can be stocked efficiently and which
should be treated differently.

| **Class** | **Criteria**   | **Description**            |
| :-------: | :------------- | :------------------------- |
| X         | CV < 0.5       | Predictable, steady demand |
| Y         | 0.5 ≤ CV < 1.0 | Moderate variability       |
| Z         | CV ≥ 1.0       | Erratic, intermittent      |

### Combined ABC-XYZ Class

The ABC class and the XYZ class are often combined into one ABC-XYZ class,
as show below:

|                     | **X (Stable)**  | **Y (Middle)**  | **Z (Erratic)** |
| ------------------: | :-------------: | :-------------: | :-------------: |
| **A (High Volume)** | AX              | AY              | AZ              |
| **B (Mid-Volume)**  | BX              | BY              | BZ              |
| **C (Low-Volume)**  | CX              | CY              | CZ              |

The combined class gives each SKU a two-letter code summarizing both its revenue
importance and its demand behavior. This drives two downstream decisions: the
**stocking policy** (whether to hold inventory proactively or produce on demand)
and the **service level target** (how often we want the item to be in stock when
a customer asks for it). A high-value, predictable item (AX) warrants both high
inventory investment and a high fill rate. A low-value, erratic item (CZ) often
costs more to stock than to order on demand.

<br />

## Stocking Policy

The stocking policy determines whether a SKU is held in inventory proactively or
procured reactively. The naming below follows retail/distribution conventions
(rather than the manufacturing "make" framing, which refers to production
scheduling — in a retail or distribution context the same logic applies to
replenishment orders):

- **MTS** = Make to Stock — items are replenished before demand arrives; safety
  stock is held to cover demand uncertainty and lead time variability.
- **MTS/review** = Make to Stock with Periodic Review — items are nominally
  stocked, but flagged for a regular review cycle. If demand falls below a
  threshold for an extended period, the item is a candidate for reclassification
  to MTO.
- **MTO** = Make to Order — items are only procured or produced after a confirmed
  order. No safety stock is held; lead time is absorbed by the customer.

The MTS/MTO decision is not binary — it is a spectrum driven by the cost of
holding inventory relative to the cost of a stockout. High-value, predictable
items earn investment in safety stock; low-value, erratic items do not. The
policy below encodes this trade-off for each segment.

- **A-Items**
  - **AX: MTS** — High revenue and predictable demand. Holding stock is
    cost-effective; a stockout on a top-revenue item is expensive in both lost
    sales and customer satisfaction.
  - **AY: MTS** — High revenue justifies the safety stock cost despite moderate
    variability. The value at risk from a stockout outweighs the holding cost.
  - **AZ: MTS** — High revenue requires reliable availability even with erratic
    demand. Stock conservatively and monitor closely; the service level target
    is reduced to 95% to manage holding cost.
- **B-Items**
  - **BX: MTS** — Predictable demand makes stocking reliable and efficient at
    mid-tier revenue.
  - **BY: MTS** — Standard MTS policy for mid-value items with moderate
    variability.
  - **BZ: MTS/review** — Erratic demand at mid-value creates holding cost risk.
    Review periodically; consider MTO if demand trends downward.
- **C-Items**
  - **CX: MTS** — Low revenue but predictable demand; standard MTS with a
    reduced service level target.
  - **CY: MTS/review** — Moderate variability at low value; a regular review
    cycle catches items drifting toward MTO-worthy behavior.
  - **CZ: MTO** — Low value and erratic demand make stocking uneconomical. Order
    only on confirmed demand, or consider discontinuing.

The table below summarizes the stocking policy by segment at a glance:

|       | **X**  | **Y**      | **Z**      |
| ----: | :----: | :--------: | :--------: |
| **A** | MTS    | MTS        | MTS        |
| **B** | MTS    | MTS        | MTS/review |
| **C** | MTS    | MTS/review | ***MTO***  |

<br />

## Service Level

The **service level** (or cycle service level) is the probability that a customer
finds the item in stock when they want it. A 99% service level means we expect to
satisfy demand from on-hand stock 99 out of 100 times; the remaining 1% results
in a stockout or backorder.

For example: if an AX item is ordered 200 times per year and we hold enough
safety stock for a 99% service level, we expect to fulfill 198 of those orders
from stock and face approximately 2 stockout events per year.

Service levels are calibrated to the revenue importance and demand predictability
of each segment — not applied uniformly across all SKUs. The logic behind the
assignments:

- **Higher service levels for A-items**: A stockout on a high-revenue SKU is
  costly in lost sales and customer goodwill. The marginal safety stock cost to
  move from 95% to 99% is justified.
- **Lower service levels for erratic items**: High variability means each
  additional percentage point of service level requires disproportionately more
  safety stock. For Z-class items the trade-off tips toward accepting a higher
  stockout risk.
- **CZ items carry no service level target**: These are MTO items — service level
  is not applicable because no stock is held.

Summarized by segment:

- **AX, AY, BX → 99%**: The combination of revenue importance and predictability
  (AX, BX) or high value alone (AY) justifies maximum fill rate commitment.
- **AZ, BY, CX → 95%**: Mid-priority segments. Either the revenue is high but
  demand is erratic (AZ), or the revenue is mid-tier with manageable variability
  (BY, CX). A 95% target balances fill rate against safety stock cost.
- **BZ, CY → 90%**: Lower-priority segments where holding excessive safety stock
  is not cost-effective given the combination of mid/low value and
  moderate-to-erratic demand.
- **CZ → N/A**: MTO policy; no service level target.

The table below maps each ABC-XYZ segment to its service level target:

|       | **X**  | **Y**  | **Z**  |
| ----: | :----: | :----: | :----: |
| **A** | 99%    | 99%    | 95%    |
| **B** | 99%    | 95%    | 90%    |
| **C** | 95%    | 90%    | N/A    |

<br />

## Conclusion

The ABC-XYZ segmentation framework provides a principled, data-driven basis for
inventory policy decisions. Rather than applying a one-size-fits-all stocking
strategy, this approach differentiates 30,490 SKU-store combinations across nine
segments — each with a tailored stocking policy and service level target.

The downstream impact is quantified in the safety stock analysis: higher service
levels require larger buffers, but the cost is concentrated in segments where it
is justified by revenue contribution. Assigning MTO policy to CZ items eliminates
holding cost for the largest segment by SKU count (~39% of combinations) while
accepting customer-facing lead time for low-value, erratic products.

Together, the stocking policy and service level assignments form the input layer
for the safety stock optimization in Sprint 5: each segment's Z-score (derived
from its service level target) is applied to the forecast error distribution to
produce per-SKU safety stock recommendations.

As of SCDF-21, that forecast-error distribution is quantified per segment in
`outputs/error_stats.csv` (bias, sigma, skew, and a Shapiro-Wilk normality
flag per ABC-XYZ cell), computed by `compute_error_stats()` in
`src/evaluation.py`.
