# Streamlit SaaS Design Patterns — Research Notes

> Research compiled March 2026 for EdgeIQ UI overhaul.
> Focus: making Streamlit apps look like professional SaaS products.

---

## 1. Custom CSS Injection Patterns

### 1a. `st.markdown()` with `unsafe_allow_html=True`

This is the primary CSS injection method and what EdgeIQ already uses via `load_css()` in `app.py`. The pattern is solid — load a single external `.css` file at boot.

```python
# Current EdgeIQ pattern (good)
def load_css():
    css_file = Path(__file__).parent / "style.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
```

**Best practices:**
- Load CSS exactly ONCE, right after `st.set_page_config()`. Multiple `<style>` tags work but add DOM bloat.
- Keep all CSS in one file (`style.css`) rather than scattering inline styles across Python files.
- Use CSS custom properties (`:root` variables) for theming — EdgeIQ already does this well.
- Avoid `!important` where possible — use specificity instead. Exception: Streamlit's internal styles sometimes require it.

**Key Streamlit CSS selectors (stable as of Streamlit 1.40+):**
```css
/* App-level */
.stApp                              /* Main app container */
[data-testid="stSidebar"]           /* Sidebar */
[data-testid="stHeader"]            /* Top header bar */
[data-testid="stToolbar"]           /* Toolbar (hamburger menu area) */
[data-testid="stStatusWidget"]      /* "Running..." spinner */
[data-testid="stDecoration"]        /* Top colored line */

/* Layout */
[data-testid="stVerticalBlock"]     /* Column content wrapper */
[data-testid="column"]              /* Individual column */
.stTabs [data-baseweb="tab-list"]   /* Tab bar container */
.stTabs [data-baseweb="tab"]        /* Individual tab */

/* Widgets */
[data-testid="stMetricValue"]       /* Metric value text */
[data-testid="stMetricLabel"]       /* Metric label text */
[data-testid="stMetricDelta"]       /* Metric delta indicator */
[data-testid="stExpander"]          /* Expander container */
[data-testid="stDataFrame"]         /* DataFrame/table */
.stButton > button                  /* Buttons */
.stSelectbox                        /* Select dropdowns */
.stSlider                           /* Sliders */

/* Containers */
[data-testid="stContainer"]         /* st.container(border=True) */
```

**WARNING:** Streamlit updates frequently break CSS selectors. The `data-testid` attributes are more stable than class names. Always prefer `[data-testid="..."]` over `.st-emotion-cache-*` classes (those hash and change every release).


### 1b. `st.html()` (Streamlit 1.33+)

`st.html()` renders raw HTML in an iframe-like sandbox. It does NOT share the page's CSS scope, so it is best for self-contained widgets (animated SVGs, embedded components). It is NOT suitable for global styling.

```python
# Good for: isolated rich components
st.html("""
<div style="background: linear-gradient(135deg, #0f172a, #1e293b);
            border-radius: 12px; padding: 24px; text-align: center;">
    <h2 style="color: #22d3ee; margin: 0;">+4.2% Edge</h2>
    <p style="color: #94a3b8; margin: 4px 0 0;">Strong signal detected</p>
</div>
""")
```

**Limitation:** `st.html` content is in a separate iframe context. It cannot interact with Streamlit widgets or session state. Use it for display-only rich cards, not for interactive elements.


### 1c. `.streamlit/config.toml` Theme

The config file sets the base theme that Streamlit's built-in widgets inherit. This is the RIGHT place for foundational colors, not CSS overrides.

```toml
# .streamlit/config.toml
[theme]
primaryColor = "#22d3ee"          # Buttons, sliders, links, active tabs
backgroundColor = "#0f172a"        # Main page background
secondaryBackgroundColor = "#1e293b" # Sidebar, containers, expanders
textColor = "#f1f5f9"             # Default text color
font = "sans serif"               # "sans serif", "serif", or "monospace"

[server]
headless = true                   # Required for Streamlit Cloud

[browser]
gatherUsageStats = false
```

**Critical insight:** If you set `config.toml` theme colors, Streamlit automatically applies them to ALL widgets — sliders, selectboxes, checkboxes, tabs, progress bars. This eliminates 60-70% of the CSS you'd otherwise need to write. EdgeIQ currently does NOT have a `config.toml` — this is the single highest-impact missing piece.

**Font options are limited** to the three built-in choices. For custom fonts like Inter (which EdgeIQ uses), you must still use the CSS `@import` approach in `style.css`.


---

## 2. Design System Approach

### 2a. Color Tokens

EdgeIQ already has a good foundation with CSS custom properties. The recommended pattern for a SaaS design system:

```css
:root {
    /* ── Semantic Tokens ── */
    --bg-primary: #0f172a;        /* Page background */
    --bg-secondary: #1e293b;      /* Cards, sidebar, containers */
    --bg-tertiary: #334155;       /* Hover states, subtle backgrounds */
    --bg-elevated: #1e293b;       /* Elevated cards (same as secondary w/ shadow) */

    --text-primary: #f1f5f9;      /* Headings, key content */
    --text-secondary: #cbd5e1;    /* Body text */
    --text-tertiary: #94a3b8;     /* Captions, labels, muted text */
    --text-inverse: #0f172a;      /* Text on light backgrounds */

    --accent: #22d3ee;            /* Primary brand accent (cyan) */
    --accent-hover: #06b6d4;      /* Accent hover state */
    --accent-muted: rgba(34, 211, 238, 0.15); /* Accent background tint */

    --success: #4ade80;           /* Win, profit, positive */
    --success-muted: rgba(74, 222, 128, 0.15);
    --warning: #facc15;           /* Lean, caution */
    --warning-muted: rgba(250, 204, 21, 0.15);
    --danger: #f87171;            /* Loss, error */
    --danger-muted: rgba(248, 113, 113, 0.15);
    --neutral: #94a3b8;           /* Pass, skip, inactive */
    --neutral-muted: rgba(148, 163, 184, 0.15);

    /* ── Borders ── */
    --border-subtle: #1e293b;
    --border-default: #334155;
    --border-strong: #475569;

    /* ── Spacing Scale ── */
    --space-xs: 4px;
    --space-sm: 8px;
    --space-md: 16px;
    --space-lg: 24px;
    --space-xl: 32px;
    --space-2xl: 48px;

    /* ── Border Radius ── */
    --radius-sm: 4px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 16px;
    --radius-full: 9999px;

    /* ── Shadows ── */
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.25);
    --shadow-lg: 0 8px 24px rgba(0,0,0,0.35);
    --shadow-glow-accent: 0 0 20px rgba(34, 211, 238, 0.15);
    --shadow-glow-success: 0 0 20px rgba(74, 222, 128, 0.15);

    /* ── Typography ── */
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    --font-size-xs: 0.75rem;      /* 12px */
    --font-size-sm: 0.875rem;     /* 14px */
    --font-size-base: 1rem;       /* 16px */
    --font-size-lg: 1.125rem;     /* 18px */
    --font-size-xl: 1.25rem;      /* 20px */
    --font-size-2xl: 1.5rem;      /* 24px */
}
```

**Key principle:** Every color, spacing, and font reference in CSS should use a variable. This enables:
1. Future light-mode support (just swap the `:root` block)
2. Consistent look across all components
3. Easier refactoring


### 2b. Typography Hierarchy

```css
/* Heading hierarchy */
h1 { font-size: 2rem; font-weight: 800; letter-spacing: -0.03em; }
h2 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em; }
h3 { font-size: 1.25rem; font-weight: 600; letter-spacing: -0.01em; }

/* Streamlit's metric values — monospace for numbers */
[data-testid="stMetricValue"] {
    font-family: var(--font-mono);
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--accent) !important;
}

/* Captions / labels */
[data-testid="stMetricLabel"],
.stCaption {
    font-size: var(--font-size-sm);
    color: var(--text-tertiary) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 500;
}
```


### 2c. Consistent Component Library (CSS Classes)

Define reusable CSS classes for common UI patterns used across all three sports:

```css
/* ── Card Component ── */
.edge-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    padding: var(--space-lg);
    transition: border-color 0.2s, box-shadow 0.2s;
}
.edge-card:hover {
    border-color: var(--accent);
    box-shadow: var(--shadow-glow-accent);
}

/* Card variants */
.edge-card--success { border-left: 3px solid var(--success); }
.edge-card--warning { border-left: 3px solid var(--warning); }
.edge-card--danger  { border-left: 3px solid var(--danger); }

/* ── Badge / Pill ── */
.edge-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 4px 10px;
    border-radius: var(--radius-full);
    font-size: var(--font-size-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    line-height: 1;
}

/* ── Stat Row (key-value pair) ── */
.edge-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-subtle);
}
.edge-stat-row:last-child { border-bottom: none; }
.edge-stat-label { color: var(--text-tertiary); font-size: var(--font-size-sm); }
.edge-stat-value { color: var(--text-primary); font-family: var(--font-mono); font-weight: 600; }

/* ── Section Header ── */
.edge-section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid var(--accent);
}
```


---

## 3. Professional Game Card / Dashboard Card Layouts

### 3a. Pattern: `st.container(border=True)` + Custom HTML Inside

The most reliable professional card pattern combines Streamlit's native container with injected HTML for the content:

```python
# Professional game card pattern
def render_game_card(home, away, home_prob, away_prob, spread, game_time, signal):
    with st.container(border=True):
        # Game header row (HTML for precise layout)
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <span style="color: var(--text-tertiary); font-size: 0.85rem;">{game_time}</span>
            <span class="edge-badge edge-badge--{signal['class']}">{signal['label']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Teams + probabilities (Streamlit columns for widget compatibility)
        left, vs, right = st.columns([5, 1, 5])
        with left:
            st.metric(home, f"{home_prob:.1f}%")
        with vs:
            st.markdown("<div style='text-align:center; padding-top:20px; color:#64748b; font-weight:700;'>VS</div>",
                       unsafe_allow_html=True)
        with right:
            st.metric(away, f"{away_prob:.1f}%")

        # Bottom bar with spread + Kelly
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-top: 8px; padding-top: 8px;
                    border-top: 1px solid var(--border-subtle); font-size: 0.85rem; color: var(--text-tertiary);">
            <span>Spread: {spread}</span>
            <span>Kelly: {signal['kelly']}</span>
        </div>
        """, unsafe_allow_html=True)
```

### 3b. Pattern: Pure HTML Card (no Streamlit widgets inside)

For display-only cards (home page, summary views), pure HTML gives complete layout control:

```python
def render_sport_card_html(sport, accuracy, features, description, badge_class):
    return f"""
    <div class="edge-card" style="height: 100%;">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="width: 48px; height: 48px; border-radius: 12px;
                        background: var(--accent-muted); display: flex;
                        align-items: center; justify-content: center; font-size: 24px;">
                {sport['icon']}
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem;">{sport['name']}</h3>
                <div style="display: flex; gap: 8px; margin-top: 4px;">
                    <span class="edge-badge {badge_class}">{accuracy}%</span>
                    <span class="edge-badge edge-badge--neutral">{features} Features</span>
                </div>
            </div>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;">
            {description}
        </p>
    </div>
    """
```

### 3c. Pattern: KPI Dashboard Row

The "4 metrics in a row" pattern is SaaS standard. Streamlit's `st.metric` handles this well but needs CSS polish:

```python
# KPI strip
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model Edge", "+4.2%")
k2.metric("Kelly %", "3.8%")
k3.metric("Bet Amount", "$38")
k4.metric("Signal", "STRONG")
```

```css
/* Make metric cards look like dashboard KPIs */
[data-testid="stMetricValue"] {
    font-family: var(--font-mono);
    font-size: 1.75rem;
    font-weight: 700;
}

/* Add subtle card treatment to metric columns */
[data-testid="column"] > [data-testid="stVerticalBlock"] {
    /* Only apply inside metric rows — use a wrapper class */
}

/* Better approach: wrap metrics in a container */
.kpi-strip [data-testid="stMetricValue"] {
    font-size: 2rem;
}
```

### 3d. Pattern: Game Result Card with Left Border Accent

```css
/* Applied to st.container(border=True) wrapping a game card */
/* Use conditional class in the markdown inside the container */
.game-card-win {
    border-left: 4px solid var(--success) !important;
    background: linear-gradient(90deg, var(--success-muted), transparent 30%);
}
.game-card-loss {
    border-left: 4px solid var(--danger) !important;
    background: linear-gradient(90deg, var(--danger-muted), transparent 30%);
}
```


---

## 4. Sidebar Design Patterns

### 4a. Structured Sidebar Sections

Professional SaaS sidebars have clear visual hierarchy:

```python
with st.sidebar:
    # Brand header
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 24px;">
        <div style="font-size: 28px;">⚡</div>
        <div>
            <div style="font-weight: 800; font-size: 1.2rem; color: #fff;">EdgeIQ</div>
            <div style="font-size: 0.75rem; color: #64748b;">NFL Terminal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Navigation / back button
    if st.button("< Back to Home", use_container_width=True):
        st.session_state['sport'] = None
        st.rerun()

    st.divider()

    # Settings section with clear header
    st.markdown("**SETTINGS**", help="Configure your betting parameters")

    bankroll = st.number_input("Bankroll ($)", min_value=100, max_value=100000,
                               value=1000, step=100, key="nfl_bankroll")

    strategy = st.selectbox("Bet Strategy",
                           ["Kelly Criterion", "Fractional Kelly", "Fixed %", "Fixed $"],
                           key="nfl_strategy")

    risk = st.select_slider("Risk Tolerance",
                           ["Conservative", "Moderate", "Aggressive"],
                           value="Moderate", key="nfl_risk")

    st.divider()

    # Info section
    with st.expander("Model Info", expanded=False):
        st.caption("26-feature stacking ensemble")
        st.caption("69.3% accuracy (2024-25 holdout)")
        st.caption("Updated: Weekly")
```

### 4b. Sidebar CSS Polish

```css
/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #020617;
    border-right: 1px solid var(--border-subtle);
    padding-top: 1rem;
}

/* Sidebar section headers */
[data-testid="stSidebar"] .stMarkdown p strong {
    color: var(--text-tertiary);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Tighter widget spacing in sidebar */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.5rem;
}

/* Sidebar select/input labels */
[data-testid="stSidebar"] label {
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
}
```


---

## 5. Tab Navigation Polish

### 5a. CSS for Professional Tabs

Streamlit's default tabs look basic. CSS transforms them:

```css
/* Tab container */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    padding: 4px;
    gap: 0;
    border-bottom: 2px solid var(--border-default);
}

/* Individual tabs */
.stTabs [data-baseweb="tab"] {
    font-family: var(--font-sans);
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-tertiary);
    padding: 10px 20px;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    border: none;
    background: transparent;
    transition: all 0.2s;
}

/* Active tab */
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    background: var(--bg-primary);
    border-bottom: 2px solid var(--accent);
}

/* Tab hover */
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary);
    background: var(--bg-tertiary);
}

/* Remove the default Streamlit tab highlight bar */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
}

/* Tab content area */
.stTabs [data-baseweb="tab-panel"] {
    padding-top: var(--space-lg);
}
```

### 5b. Alternative: Pill/Segment Navigation (Custom)

For a more modern SaaS look, some Streamlit apps use `st.radio` with `horizontal=True` styled as pills:

```python
# Custom pill navigation
nav = st.radio("", ["Game Predictor", "Backtesting", "Props", "Track Record"],
               horizontal=True, key="nav", label_visibility="collapsed")
```

```css
/* Style horizontal radio as pills */
.stRadio [role="radiogroup"] {
    display: flex;
    gap: 4px;
    background: var(--bg-secondary);
    padding: 4px;
    border-radius: var(--radius-full);
}

.stRadio [role="radiogroup"] label {
    padding: 8px 16px;
    border-radius: var(--radius-full);
    font-weight: 600;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
}

.stRadio [role="radiogroup"] label[data-checked="true"] {
    background: var(--accent);
    color: var(--text-inverse);
}
```

**Caveat:** The pill-radio approach requires careful session state management and re-renders the full page on selection (unlike tabs which are client-side). For EdgeIQ, tabs are the better choice.


---

## 6. CSS Frameworks / Patterns for Streamlit

### 6a. No External CSS Frameworks

Bootstrap, Tailwind, etc. do NOT work well with Streamlit because:
- Streamlit renders via React + Emotion CSS-in-JS internally
- External frameworks conflict with Streamlit's class names
- Tailwind utility classes cannot be applied to Streamlit-generated elements
- Bundle size is wasted since you cannot target Streamlit's DOM

**The correct approach:** Write a focused custom `style.css` with CSS custom properties (design tokens). This is what EdgeIQ already does. Expand it, do not replace it.

### 6b. Streamlit-Extras Library

`streamlit-extras` (pip installable) adds useful components:
- `card()` — pre-styled card component
- `metric_cards()` — dashboard KPI cards with colored borders
- `stylable_container()` — apply CSS classes to Streamlit containers

```python
from streamlit_extras.stylable_container import stylable_container

with stylable_container(
    key="green_card",
    css_styles="""
    {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #4ade80;
        border-radius: 12px;
        padding: 16px;
    }
    """
):
    st.metric("Win Rate", "69.3%")
    st.caption("26-feature ensemble model")
```

**Trade-off:** Adds a dependency and can make the code harder to maintain. For EdgeIQ, the manual `st.markdown(unsafe_allow_html=True)` approach is more explicit and maintainable. Consider `stylable_container` only for cases where you need to style a Streamlit native widget's wrapper element.

### 6c. `st.components.v1.html()` for Rich Components

For truly custom UI (animated charts, interactive cards):

```python
import streamlit.components.v1 as components

components.html("""
<div id="custom-widget" style="...">
    <script>
        // JavaScript here — isolated context
    </script>
</div>
""", height=200)
```

**Use sparingly.** These components live in iframes and cannot communicate with Streamlit session state (except via `Streamlit.setComponentValue()` in custom components). Best for: sparkline charts, animated number counters, custom visualizations.


---

## 7. Production Streamlit Apps — Patterns from the Wild

### 7a. Common Traits of Professional Streamlit Apps

Based on analysis of production deployments (Streamlit Gallery, community showcases, and fintech dashboards):

1. **Dark theme by default** — matches EdgeIQ. Light theme is rare in data/fintech apps.
2. **Minimal sidebar** — sidebar has settings/controls only, never main content.
3. **Hero section on landing** — big bold headline + sub-text + CTAs (EdgeIQ has this).
4. **KPI strip at top** — 3-5 key metrics in a row before detailed content.
5. **Card-based layout** — every content block is in a bordered container.
6. **Monospace numbers** — all financial/statistical values use monospace font.
7. **Color-coded signals** — green/yellow/red badges for status (EdgeIQ already has this).
8. **Subtle animations** — hover effects on cards, smooth transitions on buttons.
9. **Loading states** — `st.spinner()` with branded messages, not raw "Running...".
10. **Empty states** — when no data, show a helpful message instead of blank space.

### 7b. Anti-Patterns to Avoid

1. **Too many tabs** — max 5-6 tabs. EdgeIQ's NFL has 7, which is at the upper limit. Consider grouping.
2. **Inline styles everywhere** — moves styling into Python, hard to maintain. Consolidate into `style.css`.
3. **Over-relying on `st.columns`** — more than 5 columns looks cramped. 2-4 is the sweet spot.
4. **Missing loading states** — predictions should show spinner with "Crunching 26 features..." not silent lag.
5. **Inconsistent spacing** — mixing `st.markdown("<br>")` with `st.divider()` and `st.write("")`.
6. **Giant dataframes** — show summary metrics + expandable detail, not raw DataFrames as primary UI.


---

## 8. Actionable Recommendations for EdgeIQ

### Priority 1: Create `.streamlit/config.toml` (10 minutes, massive impact)

EdgeIQ does not have this file. Creating it will instantly fix widget colors across the entire app:

```toml
[theme]
primaryColor = "#22d3ee"
backgroundColor = "#0f172a"
secondaryBackgroundColor = "#1e293b"
textColor = "#f1f5f9"
font = "sans serif"
```

### Priority 2: Expand `style.css` Design System (30-60 minutes)

Replace current ad-hoc variables with the full design token system from Section 2a above. Add:
- Spacing scale variables
- Shadow variables
- Border radius variables
- All semantic color tokens
- Card, badge, stat-row, and section-header component classes

### Priority 3: Consolidate Inline Styles (2-3 hours)

Audit all `st.markdown(f"<div style='...'>"` calls in `app.py`, `final_app.py`, `nhl_app.py`, `mlb_app.py`. Replace inline styles with CSS classes from the design system. This will:
- Make the Python code cleaner and more readable
- Ensure consistency across sports (same card looks same everywhere)
- Make future design changes require only CSS edits, not Python edits

### Priority 4: Polish Tab Navigation (15 minutes)

Add the tab CSS from Section 5a to `style.css`. This transforms tabs from basic to professional with no Python changes.

### Priority 5: Standardize Game Card Component (1-2 hours)

Create a shared Python helper function (in a new `ui_components.py` or at the top of each app file) that renders game cards consistently across all three sports. The function should:
- Accept sport-agnostic parameters (teams, probs, spread/line, signal, kelly)
- Use CSS classes from the design system (not inline styles)
- Return consistent HTML structure

### Priority 6: Sidebar Brand + Navigation (30 minutes)

Add brand identity to sidebar across all sports:
- Logo/icon + sport label at top
- Back to home button
- Settings grouped with clear section headers
- Model info in a collapsible expander at bottom

### Priority 7: Loading States (15 minutes)

Replace bare `st.spinner("Calculating...")` with branded messages:
```python
with st.spinner("Crunching 26 features through the ensemble..."):
    result = predict(...)
```

### Priority 8: Hide Streamlit Chrome (5 minutes)

```css
/* Hide hamburger menu, "Made with Streamlit" footer, and deploy button */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
```

This removes the "this is a Streamlit app" feeling and makes it look like a standalone product.

### Priority 9: Container/Card Styling (15 minutes)

```css
/* All bordered containers get the card treatment */
[data-testid="stExpander"],
div[data-testid="stContainer"][style*="border"] {
    background: var(--bg-secondary);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-sm);
    transition: border-color 0.2s;
}

/* Container hover glow */
[data-testid="stExpander"]:hover {
    border-color: var(--border-strong);
}
```

---

## 9. Full Recommended `style.css` Structure

```
style.css
├── @import fonts
├── :root (all design tokens)
├── Base resets (.stApp, sidebar, header)
├── Typography (h1-h3, p, labels, captions)
├── Component: Metrics
├── Component: Buttons (default, primary, ghost)
├── Component: Cards/Containers
├── Component: Expanders
├── Component: Tabs
├── Component: Tables/DataFrames
├── Component: Badges (signal-lock, signal-strong, signal-lean, signal-pass)
├── Component: Stat rows
├── Component: Game cards
├── Utility: Spacing helpers (.mt-1, .mb-2, etc.)
├── Utility: Text alignment
├── Hide Streamlit chrome (MainMenu, footer, toolbar)
├── Responsive overrides (@media)
```

This keeps everything in one file, organized by layer, easy to maintain.


---

## 10. Key Takeaways

| Approach | Impact | Effort | Do It? |
|----------|--------|--------|--------|
| `.streamlit/config.toml` theme | Very High | 10 min | YES — first |
| Expand CSS design tokens | High | 30 min | YES |
| Tab CSS polish | High | 15 min | YES |
| Hide Streamlit chrome | Medium | 5 min | YES |
| Consolidate inline styles → CSS classes | High | 2-3 hrs | YES |
| Shared game card component | Medium | 1-2 hrs | YES |
| Sidebar brand + hierarchy | Medium | 30 min | YES |
| `streamlit-extras` dependency | Low | — | NO — keep it manual |
| External CSS frameworks | None | — | NO — conflicts |
| `st.html()` components | Low | — | RARELY — iframe isolation |

**Total estimated effort for full UI overhaul: 5-8 hours.**
The first 3 items (config.toml + tokens + tabs) take ~1 hour and deliver 70% of the visual improvement.
