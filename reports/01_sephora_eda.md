# Sephora Dataset — Initial EDA

## 1. Products (`product_info.csv`)

- Rows (products): **8,494**
- Columns: **27**
- Unique brands: **304**

**Missingness (key product columns):**

| column | missing | % |
|---|---|---|
| rating | 278 | 3.3% |
| reviews | 278 | 3.3% |
| ingredients | 945 | 11.1% |
| price_usd | 0 | 0.0% |
| primary_category | 0 | 0.0% |
| secondary_category | 8 | 0.1% |
| tertiary_category | 990 | 11.7% |
| highlights | 2,207 | 26.0% |
| size | 1,631 | 19.2% |

**Price (USD):**
- min 3.00 | median 35.00 | mean 51.66 | 95th 147.00 | max 1900.00

**Product rating (0-5):**
- mean 4.19 | median 4.29 | min 1.00 | max 5.00

**Top primary categories:**

| category | products |
|---|---|
| Skincare | 2,420 |
| Makeup | 2,369 |
| Hair | 1,464 |
| Fragrance | 1,432 |
| Bath & Body | 405 |
| Mini Size | 288 |
| Men | 60 |
| Tools & Brushes | 52 |
| Gifts | 4 |

- Products in a **Skincare** primary category: **2,420** (28.5%)

## 2. Reviews (`reviews_*.csv`)

- Review files: **5**
- Total reviews: **1,094,411**
- Unique products reviewed: **2,351**
- Unique reviewers (author_id): **504,823**
- Missing review_text: **1,444** (0.1%)
- Avg review length: **321** chars

**Rating distribution (reviews):**

| stars | count | % |
|---|---|---|
| 1 | 61,223 | 5.6% |
| 2 | 53,032 | 4.8% |
| 3 | 81,816 | 7.5% |
| 4 | 199,389 | 18.2% |
| 5 | 698,951 | 63.9% |

**Reviewer skin_type (self-reported):**

| skin_type | count | % |
|---|---|---|
| combination | 544,513 | 49.8% |
| dry | 185,937 | 17.0% |
| normal | 131,910 | 12.1% |
| oily | 120,494 | 11.0% |
| (missing) | 111,557 | 10.2% |

**Reviewer skin_tone (self-reported) — key for fairness:**

| skin_tone | count | % |
|---|---|---|
| light | 266,418 | 24.3% |
| fair | 208,034 | 19.0% |
| lightMedium | 196,541 | 18.0% |
| medium | 70,486 | 6.4% |
| mediumTan | 62,456 | 5.7% |
| fairLight | 56,228 | 5.1% |
| tan | 33,678 | 3.1% |
| deep | 20,601 | 1.9% |
| rich | 5,493 | 0.5% |
| olive | 1,730 | 0.2% |
| porcelain | 1,612 | 0.1% |
| dark | 522 | 0.0% |
| notSureST | 70 | 0.0% |
| ebony | 3 | 0.0% |
| (missing) | 170,539 | 15.6% |
