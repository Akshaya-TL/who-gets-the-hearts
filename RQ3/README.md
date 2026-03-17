# RQ3: Sentiment Progression Analysis

**Research Question**: Does negativity in AI art comments escalate over time (bandwagon effect) compared to human art posts?

## Hypothesis (H1)
AI art posts will show a steeper negative sentiment slope over comment position (1-50) compared to matched human art posts.

## Analysis Overview
1. **Data**: Top 50 comments per post, ordered chronologically (1=earliest, 50=latest)
2. **Sentiment**: Scored using `cardiffnlp/twitter-roberta-base-sentiment` (negative → positive numerical scores)
3. **Modeling**: Linear regression of sentiment vs comment position per post → slope distribution
4. **Comparison**: Wilcoxon signed-rank test on AI vs human slope pairs
5. **Visualization**: Mean sentiment trajectories overlaid for both conditions

## Key Files
- `rq2_sentiment_progression.ipynb` - Main analysis notebook
- `data/` - Processed HF datasets (posts, human/AI comments + dates)
- `models/` - Cached sentiment model
- `outputs/` - Regression slopes and trajectory plots

## Expected Findings
- **Null result observed**: Both conditions stayed **positive throughout** (no pile-on)
- Human art: Slightly warmer/more emotional responses
- AI art: Reserved but positive; **equalled/exceeded** human sentiment for fantasy subjects (aliens, mythology, cosmic)


**Conclusion**: Subject matter > medium. No bandwagon negativity detected.
