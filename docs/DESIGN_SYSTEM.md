# DESIGN_SYSTEM - CogniFusion

## Dashboard (dashboard.py, Streamlit)
- Layout: sidebar controls (session select, model info, thresholds), main area metrics + plots
- Colors: class colors low=green, medium=amber, high=red; confidence heatmap blue->red
- Components: metric cards (st.metric), line charts (st.line_chart), confusion matrix (st.dataframe), gate-weight bar (EEG vs eye trust)
- States: empty (no sessions -> prompt to run pipeline), loading (st.spinner during training view), error (st.error with traceback tail)

## Notebook
- Consistent section headers `## N - Title`; markdown explains why before code
- Plots: matplotlib/seaborn, figsize <= (10,5), titles on every axes, tight_layout
- Tables: pandas display() for DataFrames, CSV exports to exports/

## Rules
- No hardcoded hex in dashboard - use named constants at top of file
- Every plot must have axis labels and a title
- Accessibility: sufficient contrast, no color-only encoding (add text/hatch)

---
## For AI agents
Use only the tokens/patterns defined here. Never hardcode colors, fonts, or spacing. If a token is missing, propose one instead of inventing a value.
