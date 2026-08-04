# Human Labeling Guide & Workflow

Use this document when labeling reviews in `gold_labeling_template_part1.csv` or `gold_labeling_template_part2.csv`.

---

## Golden rule

Label from **review text only** — not star ratings (`reviews.rating` / `sentiment_rating`) and not any LLM suggestions.

---

## Label definitions

### `human_sentiment` — pick exactly ONE

| Label | Use when |
|-------|----------|
| **Positive** | Clear satisfaction or recommendation |
| **Neutral** | Factual or mixed; no strong emotion |
| **Negative** | Dissatisfaction, frustration, regret |

**Example:** 5 stars + *"It works, nothing special"* → **Neutral**

### `human_category` — pick exactly ONE

| Label | Use when |
|-------|----------|
| **Delivery issue** | Shipping, packaging, late or wrong arrival |
| **Product quality issue** | Defects, durability, performance problems |
| **Price complaint** | Too expensive, poor value for money |
| **Customer service issue** | Returns, refunds, support experience |
| **Feature request** | Suggestions, missing features, wish list |
| **Other** | General praise or no specific actionable complaint |

**Tie-break:** If two themes apply, pick the one that routes to the **most urgent team**.

### Also fill

| Column | What to write |
|--------|----------------|
| `annotator` | Your name |
| `notes` | Optional — flag ambiguous cases only |

### Allowed values (validation checks spelling exactly)

- **Sentiment:** `Positive`, `Neutral`, `Negative`
- **Category:** `Delivery issue`, `Product quality issue`, `Price complaint`, `Customer service issue`, `Feature request`, `Other`

---

## Quick examples

| Review snippet | Sentiment | Category |
|----------------|-----------|----------|
| "Love my Kindle, best purchase ever!" | Positive | Other |
| "Works fine, nothing special" | Neutral | Other |
| "Stopped working after a week" | Negative | Product quality issue |
| "Shipping took two weeks" | Negative | Delivery issue |
| "Too expensive for what you get" | Negative | Price complaint |
| "Support wouldn't help with my return" | Negative | Customer service issue |
| "Wish it had a backlight toggle" | Neutral | Feature request |

---

## Split files (50 + 50 for two annotators)

| File | Reviews | Who |
|------|---------|-----|
| `gold_labeling_template_part1.csv` / `.xlsx` | 50 | Person 1 |
| `gold_labeling_template_part2.csv` / `.xlsx` | 50 | Person 2 |

Each person labels **only their file**. Do **not** change `review_id`.

Save completed files as:
- Person 1 → `gold_labeling_completed_part1.csv`
- Person 2 → `gold_labeling_completed_part2.csv`

Full 100-row reference: `gold_labeling_template.csv`

---

## After labeling — combine, validate, merge

From the `data_mining` folder:

```bash
source .venv/bin/activate

# 1. Combine both parts (one person runs this)
python scripts/combine_human_labels.py \
  labeling/gold_labeling_completed_part1.csv \
  labeling/gold_labeling_completed_part2.csv \
  --output labeling/gold_labeling_completed.csv

# 2. Validate (must pass before merge)
python scripts/validate_human_labels.py labeling/gold_labeling_completed.csv

# 3. Merge into gold dataset
python scripts/merge_human_labels.py labeling/gold_labeling_completed.csv
```

Validation writes a report to `outputs/human_label_validation_report.txt`.

- **Exit code 0** = passed — safe to merge
- **Exit code 1** = failed — fix errors and re-run

---

## Speed tips

- ~2 minutes per review → ~1.5–2 hours for 50 reviews
- Don't overthink edge cases; use `notes` and move on
- Hide or ignore columns you are not filling (`sentiment_rating` is reference only)
