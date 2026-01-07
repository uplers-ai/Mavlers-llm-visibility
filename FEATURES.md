# 🚀 New Features Documentation

## Overview
This document describes the new features added to `visibility_audit3.0.py` for tracking goals and monthly comparisons.

---

## 📋 Table of Contents
1. [Goals System](#goals-system)
2. [Monthly Comparison](#monthly-comparison)
3. [Dashboard Enhancements](#dashboard-enhancements)
4. [Email Notifications](#email-notifications)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)

---

## 🎯 Goals System

### Description
The Goals System allows you to set target visibility scores for each LLM and track your progress toward achieving them. The system automatically calculates progress percentages and provides visual indicators in the dashboard and email reports.

### Features
- **Per-LLM Targets**: Set individual visibility score targets for each LLM (ChatGPT, Claude, Gemini, Grok, Perplexity)
- **Overall Targets**: Set overall visibility score and ranking targets
- **Progress Tracking**: Real-time calculation of progress percentage toward each goal
- **Status Indicators**: Color-coded status (🟢 Achieved, 🟡 In Progress, 🔴 Far)
- **Visual Progress Bars**: Graphical representation of goal progress in the dashboard

### Configuration
Located at the top of the script (lines 135-151):

```python
GOALS = {
    "overall_visibility_score": 35,  # Target overall visibility (average of all LLMs)
    "overall_rank": 10,              # Target to be in top 10 ranking
    "by_llm": {
        "ChatGPT": 40,      # Target 40% visibility on ChatGPT
        "Claude": 25,       # Target 25% visibility on Claude
        "Gemini": 40,       # Target 40% visibility on Gemini
        "Grok": 25,         # Target 25% visibility on Grok
        "Perplexity": 40,   # Target 40% visibility on Perplexity
    },
}
```

### How It Works
1. **Goal Calculation**: After each audit, the script calculates:
   - Current visibility score for each LLM
   - Progress percentage: `(current / target) × 100`
   - Remaining percentage needed to reach goal
   - Status: "achieved" (≥100%), "in_progress" (≥50%), or "far" (<50%)

2. **Dashboard Display**: 
   - Progress bars showing visual progress
   - Color-coded cards (green/yellow/red)
   - Summary showing "X/Y LLMs achieved target"
   - Individual goal cards with current vs target percentages

3. **Console Output**:
   ```
   🎯 GOALS PROGRESS: 2/5 LLMs achieved target
      ✅ ChatGPT: 42% / 40% target (105% progress)
      🟡 Claude: 18% / 25% target (72% progress)
      🔴 Grok: 5% / 25% target (20% progress)
   ```

### Status Definitions
- **🟢 Achieved**: Current score ≥ Target score
- **🟡 In Progress**: Current score ≥ 50% of target
- **🔴 Far**: Current score < 50% of target

---

## 📅 Monthly Comparison

### Description
The Monthly Comparison feature aggregates audit data by month and provides side-by-side comparisons of visibility scores, mentions, and rankings across the last 3 months (configurable).

### Features
- **Multi-Month Aggregation**: Automatically groups audit results by month
- **Average Calculations**: Computes average visibility scores and rankings per month
- **Mention Tracking**: Tracks total mentions per LLM per month
- **Change Indicators**: Shows month-over-month changes with arrows (↑/↓)
- **Visual Table**: Clean, organized table format in the dashboard

### Configuration
```python
MONTHS_TO_COMPARE = 3  # Compare last 3 months
```

### How It Works
1. **Data Aggregation**: 
   - Groups all audit results by month (YYYY-MM format)
   - Calculates averages for visibility scores and rankings
   - Sums total mentions per LLM per month

2. **Comparison Table**:
   - Shows each LLM's performance across months
   - Displays visibility percentage and total mentions
   - Calculates and shows changes between months
   - Includes overall row with aggregated metrics

3. **Example Output**:
   ```
   | LLM        | Nov 2025 | Dec 2025 | Jan 2026 | Change |
   |------------|----------|----------|----------|--------|
   | ChatGPT    | 35% (45) | 38% (52) | 42% (62) | ↑4%    |
   | Claude     | 22% (28) | 25% (32) | 18% (28) | ↓7%    |
   | Perplexity | 45% (55) | 50% (60) | 55% (71) | ↑5%    |
   ```

### Data Requirements
- Requires at least 2 months of historical audit data
- Automatically uses the most recent N months (where N = `MONTHS_TO_COMPARE`)
- If less than 2 months available, the section won't appear

---

## 🎨 Dashboard Enhancements

### Goals Progress Section
**Location**: Appears after the Weekly/Monthly Changes section

**Features**:
- Summary header showing "X/Y LLMs achieved target"
- Grid layout with goal cards for each LLM
- Progress bars with color-coded fills
- Status badges (✓ Achieved / In Progress / Needs Work)
- Current vs Target percentage display
- Remaining percentage to goal

**Visual Design**:
- Green border: Goal achieved
- Yellow border: In progress (≥50%)
- Red border: Far from goal (<50%)
- Animated progress bars

### Monthly Comparison Section
**Location**: Appears after Goals Progress section

**Features**:
- Responsive table layout
- Month headers (e.g., "Nov 2025", "Dec 2025", "Jan 2026")
- Per-LLM rows with visibility % and mention counts
- Change column showing month-over-month differences
- Overall row with aggregated metrics
- Color-coded change indicators (green for positive, red for negative)

**Table Columns**:
1. **LLM**: Model name
2. **Month Columns**: Visibility % and total mentions in parentheses
3. **Change**: Arrow indicator (↑/↓) with percentage and mention change

---

## 📧 Email Notifications

### Goals Section in Email
**Location**: Appears after the "Performance by LLM" table

**Features**:
- Summary header: "🎯 GOALS PROGRESS (X/Y Achieved)"
- Table format with:
  - Status icon (✅/🟡/🔴)
  - LLM name
  - Current % / Target %
  - Progress percentage
- Color-coded status indicators

**Example Email Section**:
```
🎯 GOALS PROGRESS (2/5 Achieved)
✅ ChatGPT    42% / 40%    105%
🟡 Claude     18% / 25%    72%
🔴 Grok       5% / 25%     20%
🟡 Gemini     32% / 40%    80%
✅ Perplexity 45% / 40%    113%
```

---

## ⚙️ Configuration

### Setting Goals
Edit the `GOALS` dictionary in the script:

```python
GOALS = {
    "overall_visibility_score": 50,  # Change overall target
    "overall_rank": 5,               # Change rank target
    "by_llm": {
        "ChatGPT": 50,      # Increase ChatGPT target
        "Claude": 30,       # Increase Claude target
        "Gemini": 45,       # Adjust Gemini target
        "Grok": 30,         # Adjust Grok target
        "Perplexity": 50,   # Increase Perplexity target
    },
}
```

### Adjusting Comparison Period
Change the number of months to compare:

```python
MONTHS_TO_COMPARE = 6  # Compare last 6 months instead of 3
```

---

## 📊 Usage Examples

### Example 1: Checking Goal Progress
After running an audit, check the console output:
```
🎯 GOALS PROGRESS: 2/5 LLMs achieved target
   ✅ ChatGPT: 42% / 40% target (105% progress)
   🟡 Claude: 18% / 25% target (72% progress)
```

### Example 2: Viewing Monthly Trends
The dashboard will show a table like:
```
| LLM        | Oct 2025 | Nov 2025 | Dec 2025 | Change |
|------------|----------|----------|----------|--------|
| ChatGPT    | 35% (45) | 38% (52) | 42% (62) | ↑4%    |
```

### Example 3: Customizing Targets
To set more aggressive targets:
```python
GOALS = {
    "by_llm": {
        "ChatGPT": 60,      # Aim for 60% on ChatGPT
        "Perplexity": 55,   # Aim for 55% on Perplexity
    },
}
```

---

## 🔧 Technical Details

### Functions Added

1. **`calculate_goal_progress(analysis: dict) -> dict`**
   - Calculates progress toward all goals
   - Returns structured data with status, percentages, and remaining values
   - Location: Lines ~420-500

2. **`get_monthly_aggregates(months: int = 3) -> list`**
   - Aggregates audit data by month
   - Calculates averages and totals
   - Returns list of monthly summaries
   - Location: Lines ~500-600

3. **`calculate_monthly_changes(monthly_aggregates: list) -> dict`**
   - Calculates month-over-month changes
   - Provides direction indicators (up/down/same)
   - Location: Lines ~600-700

### Data Storage
- Goal progress is saved in `audit_results.json` under the `goal_progress` key
- Monthly aggregates are calculated on-the-fly from archived results
- No additional storage required - uses existing archive files

### Performance
- Goal calculation: O(n) where n = number of LLMs
- Monthly aggregation: O(m × n) where m = number of audits, n = number of LLMs
- Both operations are fast and don't significantly impact audit runtime

---

## 📈 Benefits

1. **Clear Visibility**: Instantly see which LLMs are meeting targets
2. **Trend Analysis**: Understand month-over-month performance changes
3. **Goal Setting**: Set realistic targets and track progress
4. **Data-Driven Decisions**: Make informed decisions based on historical trends
5. **Automated Reporting**: Goals and comparisons included in every audit report

---

## 🐛 Troubleshooting

### Goals Not Showing
- **Issue**: Goals section doesn't appear in dashboard
- **Solution**: Ensure `goal_progress` is calculated and passed to `generate_html_dashboard()`

### Monthly Comparison Empty
- **Issue**: Monthly comparison table is empty
- **Solution**: 
  - Check that you have at least 2 months of audit data in the `archives/` folder
  - Verify archive files are named correctly: `audit_results_YYYY-MM-DD.json`
  - Increase `MONTHS_TO_COMPARE` if you want more months

### Goals Status Incorrect
- **Issue**: Goal status shows "Far" when it should be "Achieved"
- **Solution**: Check that the current visibility score is actually ≥ target score. The calculation is: `(current / target) × 100`

---

## 📝 Notes

- Goals are evaluated after each audit run
- Monthly aggregates are recalculated each time (no caching)
- Historical data is read from archived JSON files
- Goals can be updated at any time - changes take effect on the next audit

---

## 🔄 Future Enhancements (Potential)

- Goal alerts when targets are achieved
- Custom goal periods (weekly, quarterly, etc.)
- Goal history tracking
- Export goals data to CSV/Excel
- Goal achievement notifications

---

**Last Updated**: January 2026  
**Version**: 3.0  
**Script**: `visibility_audit3.0.py`

