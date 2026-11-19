"""
Supply Chain Analytics - Inventory Decision Agent App
Provides dual interfaces:
1. Rich Interactive CLI / Terminal Runner
2. Streamlit Web Dashboard (via `streamlit run app.py`)

Supports all 5 core inventory models:
- Economic Order Quantity (EOQ)
- Quantity Discount Analysis
- Safety Stock & Reorder Point (Continuous Review)
- Fixed-Period Review (Periodic Review / Order-Up-To)
- Newsvendor Single-Period Optimization
"""

import argparse
import os
import sys
from typing import Any, Dict, List

from inventory_agent import InventoryAgent, InventoryModelType

BENCHMARK_SCENARIOS = [
    {
        "model": "EOQ",
        "title": "1. Economic Order Quantity (EOQ) Standard Benchmark",
        "question": "Calculate the Economic Order Quantity for an industrial motor part. Annual demand is 10,000 units, order setup cost is $50 per order, and unit holding cost is $2 per unit per year. Lead time is 5 days (assume 250 working days/year).",
    },
    {
        "model": "QUANTITY_DISCOUNT",
        "title": "2. Quantity Discount Tier Evaluation",
        "question": "A distributor has annual demand of 1,000 units and order cost of $40. Holding cost rate is 20% of unit price. The vendor offers price breaks: 1 to 499 units at $5.00, 500 to 999 units at $4.50, and 1000+ units at $3.90. What is the optimal order quantity?",
    },
    {
        "model": "SAFETY_STOCK_ROP",
        "title": "3. Safety Stock & Reorder Point (Continuous Review)",
        "question": "A retail pharmacy sells an average of 50 units per day with a standard deviation of demand of 10 units. Supplier lead time is 4 days with lead time standard deviation of 1 day. Target cycle service level is 95%. Unit holding cost is $1.50/year. What is the safety stock and reorder point?",
    },
    {
        "model": "FIXED_PERIOD_REVIEW",
        "title": "4. Fixed-Period Review (Periodic Order-Up-To)",
        "question": "We review inventory every 14 days. Lead time is 7 days. Average daily demand is 20 units with a standard deviation of 5 units. Target service level is 95%. Current inventory on hand is 120 units, on order is 80 units, and backorders are 10 units. How much should we order?",
    },
    {
        "model": "NEWSVENDOR",
        "title": "5. Newsvendor Perishable Inventory Optimization",
        "question": "A boutique sells designer winter coats for $100 each. The wholesale purchase cost is $40 per unit. Any unsold coats at the end of the season can be salvaged for $10. Expected seasonal demand is 500 units with a standard deviation of 100 units. What is the optimal order quantity and expected profit?",
    },
]


# ===========================================================================
# Terminal / CLI Mode Implementation
# ===========================================================================
def run_cli_single(agent: InventoryAgent, question: str, console: Any):
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.table import Table

    console.print(Panel(f"[bold cyan]Customer Question:[/bold cyan]\n{question}", title="📥 Input Query", border_style="cyan"))

    response = agent.process_question(question)

    # Pipeline Steps Display
    pipeline_table = Table(title="🔄 Agent Decision Pipeline Execution", show_header=True, header_style="bold magenta")
    pipeline_table.add_column("Pipeline Stage", style="bold")
    pipeline_table.add_column("Agent Action / Result", style="green")

    pipeline_table.add_row("1. Identify Model", f"[yellow]{response.model_name}[/yellow] ({response.model_type})")
    
    # Formatted inputs
    inputs_str = ", ".join(f"{k}={v}" for k, v in response.raw_inputs.items() if v is not None)
    pipeline_table.add_row("2. Extract Inputs", inputs_str or "[dim]None[/dim]")

    if response.error:
        pipeline_table.add_row("3. Execution Status", f"[bold red]FAILED: {response.error}[/bold red]")
        console.print(pipeline_table)
        console.print(Panel(f"[bold red]Error:[/bold red] {response.explanation}", border_style="red"))
        return

    pipeline_table.add_row("3. Select Python Tool", f"inventory_tools.{agent.select_tool(response.model_type)[1].split('.')[-1]}")
    pipeline_table.add_row("4. Calculate", "Executed numerical algorithms in pure Python")
    status_color = "green" if response.verification_passed else "yellow"
    pipeline_table.add_row("5. Verify", f"[{status_color}]Passed ({len(response.verification_notes)} validation checks)[/{status_color}]")
    pipeline_table.add_row("6. Explain", "Generated executive analytics briefing")

    console.print(pipeline_table)
    console.print()

    # Executive Briefing
    console.print(Panel(Markdown(response.explanation), title="📊 Executive Supply Chain Report", border_style="green"))


def run_cli_interactive(agent: InventoryAgent):
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt

    console = Console()
    console.print(Panel.fit(
        "[bold green]Supply Chain Analytics - Inventory Decision Agent[/bold green]\n"
        "[cyan]Natural Language Inventory Optimization Engine[/cyan]\n"
        "[dim]Supports: EOQ | Quantity Discount | Safety Stock & ROP | Fixed-Period Review | Newsvendor[/dim]\n\n"
        "[yellow]Tip:[/yellow] Run '[bold white]streamlit run app.py[/bold white]' to launch the interactive Web Dashboard!",
        border_style="green",
    ))

    while True:
        console.print("\n[bold]Select an option:[/bold]")
        console.print("  [cyan]1[/cyan] - Type a custom natural-language inventory question")
        console.print("  [cyan]2[/cyan] - Run standard 5-Model Benchmark Suite")
        console.print("  [cyan]3[/cyan] - Select from preset supply chain case studies")
        console.print("  [cyan]q[/cyan] - Exit")

        choice = Prompt.ask("\nEnter choice", choices=["1", "2", "3", "q"], default="1")

        if choice == "q":
            console.print("[dim]Exiting Inventory Decision Agent. Goodbye![/dim]")
            break

        elif choice == "1":
            q = Prompt.ask("\n[bold cyan]Enter your inventory question[/bold cyan]")
            if q.strip():
                run_cli_single(agent, q.strip(), console)

        elif choice == "2":
            console.print("\n[bold yellow]Running Complete 5-Model Supply Chain Benchmark Suite...[/bold yellow]\n")
            for idx, item in enumerate(BENCHMARK_SCENARIOS, 1):
                console.rule(f"[bold blue]{item['title']}[/bold blue]")
                run_cli_single(agent, item["question"], console)
                console.print()

        elif choice == "3":
            console.print("\n[bold]Preset Case Studies:[/bold]")
            for i, b in enumerate(BENCHMARK_SCENARIOS, 1):
                console.print(f"  [cyan]{i}[/cyan] - {b['title']}")
            sub_choice = Prompt.ask("Choose a case study (1-5)", choices=["1", "2", "3", "4", "5"], default="1")
            selected = BENCHMARK_SCENARIOS[int(sub_choice) - 1]
            run_cli_single(agent, selected["question"], console)


# ===========================================================================
# CSV Multicriteria Inventory Classification Helper
# ===========================================================================
REQUIRED_CLASSIFICATION_COLUMNS = ["Item Number", "Item Name", "UoM", "Sales ($)", "Hits"]
PARETO_SCORE_MAP = {"A": 4, "B": 3, "C": 2, "D": 1}


def _classify_pareto_series(series: "pd.Series") -> "pd.Series":
    """
    Sort items descending and assign Pareto classification thresholds:
    A: 0 - 60%
    B: 61 - 80%
    C: 81 - 90%
    D: 91 - 100%
    """
    import pandas as pd

    sorted_s = series.sort_values(ascending=False)
    total = float(sorted_s.sum())
    if total <= 0:
        return pd.Series("D", index=series.index)

    cum_pct = (sorted_s.cumsum() / total) * 100.0

    classes = []
    for idx, (_orig_idx, val) in enumerate(cum_pct.items()):
        if idx == 0 and val > 60.0:
            # Single top driver is always at least class A
            classes.append("A")
        elif val <= 60.0:
            classes.append("A")
        elif val <= 80.0:
            classes.append("B")
        elif val <= 90.0:
            classes.append("C")
        else:
            classes.append("D")

    return pd.Series(classes, index=sorted_s.index).reindex(series.index)


def _assign_final_class(score: float) -> str:
    """
    Assign Final Inventory Class based on Combined Weighted Score:
    A: 4.00 - 3.25
    B: 3.24 - 2.50
    C: 2.49 - 1.75
    D: < 1.75
    """
    s = round(score, 2)
    if s >= 3.25:
        return "A"
    elif s >= 2.50:
        return "B"
    elif s >= 1.75:
        return "C"
    else:
        return "D"


def classify_inventory_dataframe(df: "pd.DataFrame") -> Dict[str, Any]:
    """
    Perform multicriteria inventory classification on the uploaded dataframe:
    Step 1 - Classify Sales and POS Hits Separately (Pareto: A=0-60%, B=61-80%, C=81-90%, D=91-100%)
             Convert to numerical scores (A=4, B=3, C=2, D=1).
    Step 2 - Combined Weighted Score = 0.65 * (Sales Score) + 0.35 * (Hits Score).
    Step 3 - Final Inventory Class assignment (A: 4.00-3.25, B: 3.24-2.50, C: 2.49-1.75, D: <1.75).
    Produces Required Classification Summary: Item counts and total Sales ($) by Category.
    """
    import pandas as pd

    # Check for column presence (accept 'Sales' as alias for 'Sales ($)' if needed)
    df_cols = list(df.columns)
    sales_col = "Sales ($)" if "Sales ($)" in df_cols else ("Sales" if "Sales" in df_cols else None)

    missing_cols = []
    for col in ["Item Number", "Item Name", "UoM", "Hits"]:
        if col not in df_cols:
            missing_cols.append(col)
    if sales_col is None:
        missing_cols.append("Sales ($)")

    if missing_cols:
        return {
            "valid": False,
            "missing_columns": missing_cols,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": list(df.columns),
            "df": df,
        }

    cdf = df.copy()

    # Normalize sales column name if uploaded as 'Sales'
    if sales_col != "Sales ($)":
        cdf["Sales ($)"] = cdf[sales_col]

    # Clean numeric types in case numbers were formatted with currency or commas
    if not pd.api.types.is_numeric_dtype(cdf["Sales ($)"]):
        cdf["Sales ($)"] = pd.to_numeric(
            cdf["Sales ($)"].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
            errors="coerce",
        ).fillna(0.0)

    if not pd.api.types.is_numeric_dtype(cdf["Hits"]):
        cdf["Hits"] = pd.to_numeric(
            cdf["Hits"].astype(str).str.replace(r"[,]", "", regex=True).str.strip(),
            errors="coerce",
        ).fillna(0.0)

    # Step 1: Classify Sales and Hits separately
    cdf["Sales Class"] = _classify_pareto_series(cdf["Sales ($)"])
    cdf["Hits Class"] = _classify_pareto_series(cdf["Hits"])

    # Convert classes to numerical scores
    cdf["Sales Score"] = cdf["Sales Class"].map(PARETO_SCORE_MAP).fillna(1).astype(int)
    cdf["Hits Score"] = cdf["Hits Class"].map(PARETO_SCORE_MAP).fillna(1).astype(int)

    # Step 2: Calculate Combined Weighted Score (65% Sales, 35% Hits)
    cdf["Combined Weighted Score"] = (0.65 * cdf["Sales Score"] + 0.35 * cdf["Hits Score"]).round(2)

    # Step 3: Assign Final Inventory Class
    cdf["Final Inventory Class"] = cdf["Combined Weighted Score"].apply(_assign_final_class)

    # Required Classification Summary (A, B, C, D counts and total sales)
    summary_rows = []
    total_sales = float(cdf["Sales ($)"].sum())
    total_items = len(cdf)

    for cat in ["A", "B", "C", "D"]:
        cat_df = cdf[cdf["Final Inventory Class"] == cat]
        cnt = len(cat_df)
        cat_sales = float(cat_df["Sales ($)"].sum())
        pct_items = (cnt / total_items * 100.0) if total_items > 0 else 0.0
        pct_sales = (cat_sales / total_sales * 100.0) if total_sales > 0 else 0.0

        summary_rows.append({
            "Final Inventory Class": cat,
            "Number of Items": cnt,
            "% of Items": round(pct_items, 1),
            "Total Sales ($)": round(cat_sales, 2),
            "% of Total Sales": round(pct_sales, 1),
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "valid": True,
        "missing_columns": [],
        "num_rows": len(cdf),
        "num_columns": len(cdf.columns),
        "column_names": list(df.columns),
        "classification_summary": summary_df,
        "classified_df": cdf,
        "total_items": total_items,
        "total_sales": total_sales,
    }


# ===========================================================================
# CSV Inventory Data Analysis Helper
# ===========================================================================
REQUIRED_CSV_COLUMNS = ["Product", "Category", "Units", "Unit Cost"]


def analyze_inventory_csv(df: "pd.DataFrame") -> Dict[str, Any]:
    """
    Validate and analyze an uploaded inventory CSV dataset using pandas.
    Calculates:
    - total Units
    - average Unit Cost
    - product with highest Units
    - total Units grouped by Category
    - calculated column: Inventory Value = Units * Unit Cost
    - total Inventory Value across all products
    """
    import pandas as pd

    # 1. Validation
    missing_columns = [col for col in REQUIRED_CSV_COLUMNS if col not in df.columns]
    if missing_columns:
        return {
            "valid": False,
            "missing_columns": missing_columns,
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "column_names": list(df.columns),
            "df": df,
        }

    # Work on a copy to avoid mutating external references
    analysis_df = df.copy()

    # Clean numeric types if strings with currency symbols or commas were uploaded
    if not pd.api.types.is_numeric_dtype(analysis_df["Units"]):
        analysis_df["Units"] = pd.to_numeric(
            analysis_df["Units"].astype(str).str.replace(r"[,]", "", regex=True).str.strip(),
            errors="coerce",
        )
    if not pd.api.types.is_numeric_dtype(analysis_df["Unit Cost"]):
        analysis_df["Unit Cost"] = pd.to_numeric(
            analysis_df["Unit Cost"].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip(),
            errors="coerce",
        )

    # 2. Data Analysis using pandas
    total_units = float(analysis_df["Units"].sum()) if len(analysis_df) > 0 else 0.0
    avg_unit_cost = float(analysis_df["Unit Cost"].mean()) if len(analysis_df) > 0 else 0.0

    if len(analysis_df) > 0 and not analysis_df["Units"].isna().all():
        max_idx = analysis_df["Units"].idxmax()
        top_product = str(analysis_df.loc[max_idx, "Product"])
        top_product_units = float(analysis_df.loc[max_idx, "Units"])
    else:
        top_product = "N/A"
        top_product_units = 0.0

    # Category Grouping
    category_summary = (
        analysis_df.groupby("Category", as_index=False)["Units"]
        .sum()
        .rename(columns={"Units": "Total Units"})
    )

    # Calculated column: Inventory Value = Units * Unit Cost
    analysis_df["Inventory Value"] = analysis_df["Units"] * analysis_df["Unit Cost"]
    total_inventory_value = float(analysis_df["Inventory Value"].sum()) if len(analysis_df) > 0 else 0.0

    return {
        "valid": True,
        "missing_columns": [],
        "num_rows": len(analysis_df),
        "num_columns": len(analysis_df.columns),
        "column_names": list(df.columns),
        "total_units": total_units,
        "average_unit_cost": avg_unit_cost,
        "top_product": top_product,
        "top_product_units": top_product_units,
        "category_summary": category_summary,
        "total_inventory_value": total_inventory_value,
        "df_with_inventory_value": analysis_df,
    }


# ===========================================================================
# Streamlit Web UI Mode Implementation
# ===========================================================================
def run_streamlit_app():
    import streamlit as st
    import pandas as pd

    st.set_page_config(
        page_title="Supply Chain Analytics - Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.markdown(
        "Autonomous operations research agent that identifies inventory problem types from "
        "natural-language inquiries, calculates exact closed-form numerical solutions in pure Python, "
        "verifies mathematical invariants, and outputs executive-ready decisions."
    )

    agent = InventoryAgent()

    # Sidebar
    st.sidebar.header("🎯 Inventory Decision Pipeline")
    st.sidebar.markdown(
        """
        **Architecture Pipeline:**
        1. 📥 **Customer Question**
        2. 🔍 **Identify Model**
        3. 📐 **Extract Inputs**
        4. ⚙️ **Select Python Tool**
        5. 🧮 **Calculate**
        6. ✅ **Verify Invariants**
        7. 📊 **Explain & Recommend**
        """
    )
    st.sidebar.divider()
    st.sidebar.subheader("Supported Operations Models")
    st.sidebar.markdown(
        """
        - **EOQ**: Batch size, holding vs ordering parity
        - **Quantity Discount**: All-units cutoff analysis
        - **Safety Stock & ROP**: Multi-source stochastic variance
        - **Fixed-Period Review**: (T, S) vulnerability window
        - **Newsvendor**: Single-period critical fractile
        """
    )

    # Data Upload section as the last section in the sidebar
    st.sidebar.divider()
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

    # Preset Selector
    preset_names = ["-- Custom Natural Language Question --"] + [b["title"] for b in BENCHMARK_SCENARIOS]
    selected_preset = st.selectbox("Load a canonical supply chain scenario or write your own:", preset_names)

    default_text = ""
    if selected_preset != "-- Custom Natural Language Question --":
        for b in BENCHMARK_SCENARIOS:
            if b["title"] == selected_preset:
                default_text = b["question"]
                break

    # Question Input
    user_question = st.text_area(
        "Enter your natural-language inventory question:",
        value=default_text or "What is the EOQ for an item with annual demand of 12000 units, ordering cost of $75 per order, unit purchase cost of $15, and annual holding rate of 18%?",
        height=110,
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🚀 Analyze & Optimize", type="primary")

    if analyze_btn or default_text:
        with st.spinner("Agent running decision pipeline..."):
            response = agent.process_question(user_question)

        # Pipeline steps indicator
        st.subheader("🔄 Agent Execution Pipeline")
        p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
        with p_col1:
            st.metric("1. Model Identified", response.model_type)
        with p_col2:
            st.metric("2. Inputs Parsed", f"{len(response.raw_inputs)} parameters")
        with p_col3:
            st.metric("3. Python Tool", response.model_name.split()[0])
        with p_col4:
            st.metric("4. Verification", "PASSED ✅" if response.verification_passed else "FLAGGED ⚠️")
        with p_col5:
            st.metric("5. Decision Ready", "YES 🎯")

        st.divider()

        if response.error:
            st.error(f"❌ {response.error}")
            st.info(response.explanation)
        else:
            # Display KPIs if structured metrics exist
            metrics = response.structured_metrics
            if response.model_type == InventoryModelType.EOQ:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Optimal Order Q*", f"{metrics.get('optimal_order_quantity', 0):,.0f} units")
                k2.metric("Orders / Year", f"{metrics.get('orders_per_year', 0):.1f}")
                k3.metric("Total Inventory Cost", f"${metrics.get('total_inventory_cost', 0):,.2f}")
                k4.metric("Cycle Time", f"{metrics.get('cycle_time_days', 0):.1f} days")

            elif response.model_type == InventoryModelType.QUANTITY_DISCOUNT:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Optimal Order Q*", f"{metrics.get('optimal_order_quantity', 0):,.0f} units")
                k2.metric("Optimal Tier Price", f"${metrics.get('optimal_price', 0):.2f}")
                k3.metric("Total Acquisition Cost", f"${metrics.get('minimum_total_acquisition_cost', 0):,.2f}")
                k4.metric("Net Annual Savings", f"${metrics.get('net_annual_savings', 0):,.2f}")

                # Tier breakdown dataframe
                if "tier_evaluations" in metrics:
                    st.write("### 📋 Discount Tiers Breakdown")
                    st.dataframe(pd.DataFrame(metrics["tier_evaluations"]), use_container_width=True)

            elif response.model_type == InventoryModelType.SAFETY_STOCK_ROP:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Reorder Point (ROP)", f"{metrics.get('reorder_point', 0):,.1f} units")
                k2.metric("Safety Stock Buffer", f"{metrics.get('safety_stock', 0):,.1f} units")
                k3.metric("Service Level (CSL)", metrics.get("service_level", "95%"))
                k4.metric("Z-Score", f"{metrics.get('z_score', 0):.3f}")

            elif response.model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Replenishment Q", f"{metrics.get('recommended_order_quantity', metrics.get('order_quantity', 0)):,.0f} units")
                k2.metric("Order-Up-To Level (S)", f"{metrics.get('target_inventory', metrics.get('target_order_up_to_level_S', 0)):,.1f} units")
                k3.metric("Inventory Position", f"{metrics.get('inventory_position', 0):,.0f} units")
                k4.metric("Protection Interval", f"{metrics.get('protection_period', metrics.get('protection_interval_T_plus_L', 0)):.1f} periods")

            elif response.model_type == InventoryModelType.NEWSVENDOR:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Optimal Stocking Q*", f"{metrics.get('recommended_order_quantity', metrics.get('optimal_order_quantity', 0)):,.0f} units")
                k2.metric("Critical Ratio (CR)", f"{metrics.get('critical_ratio', 0) * 100:.2f}%")
                k3.metric("Expected Profit", f"${metrics.get('expected_profit', 0):,.2f}")
                k4.metric("Expected Sales", f"{metrics.get('expected_sales', 0):,.1f} units")

            # Explanation markdown (contains 8 student verification sections)
            st.markdown(response.explanation)

            # Verification checklist & raw structured payload
            with st.expander("🎓 Student Verification Data (Raw Structured JSON)", expanded=False):
                st.json(metrics)

    # -----------------------------------------------------------------------
    # CSV Data Upload, Classification & Analysis Section
    # -----------------------------------------------------------------------
    if uploaded_file is not None:
        st.divider()
        st.header("📊 Uploaded Inventory Dataset Operations")

        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {e}")
            raw_df = None

        if raw_df is not None:
            # Display: number of rows, number of columns, column names
            st.subheader("📋 Dataset Overview")
            meta_col1, meta_col2, meta_col3 = st.columns([1, 1, 3])
            meta_col1.metric("Number of Rows", f"{len(raw_df):,}")
            meta_col2.metric("Number of Columns", f"{len(raw_df.columns):,}")
            meta_col3.markdown(f"**Column Names:** {', '.join(f'`{c}`' for c in raw_df.columns)}")

            # Detect which inventory capability matches the uploaded columns
            is_classification_dataset = any(
                c in raw_df.columns for c in ["Sales ($)", "Hits", "Item Number", "Item Name", "UoM"]
            )
            is_units_analysis_dataset = any(
                c in raw_df.columns for c in ["Product", "Category", "Units", "Unit Cost"]
            )

            # ---------------------------------------------------------------
            # 1. Multicriteria Inventory Classification
            # ---------------------------------------------------------------
            if is_classification_dataset:
                res_class = classify_inventory_dataframe(raw_df)

                if not res_class["valid"]:
                    st.error(f"❌ Missing required classification columns: {', '.join(res_class['missing_columns'])}")
                    st.info("Multicriteria inventory classification requires: **Item Number**, **Item Name**, **UoM**, **Sales ($)**, and **Hits**.")
                    st.subheader("📄 Uploaded Dataframe")
                    st.dataframe(raw_df, use_container_width=True)
                else:
                    st.divider()
                    st.subheader("🏷️ Multicriteria Inventory Classification Summary")
                    st.markdown(
                        "**Decision Matrix Logic:**\n"
                        "- **Step 1 (Separate Pareto):** Sales & Hits ranked descending: A (0-60%), B (61-80%), C (81-90%), D (91-100%). Scores: A=4, B=3, C=2, D=1.\n"
                        "- **Step 2 (Combined Score):** `0.65 × (Sales Score) + 0.35 × (Hits Score)`.\n"
                        "- **Step 3 (Final Class):** A (4.00–3.25), B (3.24–2.50), C (2.49–1.75), D (< 1.75)."
                    )

                    # Required Classification Summary Table: Counts and Total Sales ($) by Category A, B, C, D
                    summary_df = res_class["classification_summary"]
                    st.dataframe(
                        summary_df,
                        column_config={
                            "Final Inventory Class": st.column_config.TextColumn("Final Class"),
                            "Number of Items": st.column_config.NumberColumn("Number of Items", format="%d"),
                            "% of Items": st.column_config.NumberColumn("% of Items", format="%.1f%%"),
                            "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%.2f"),
                            "% of Total Sales": st.column_config.NumberColumn("% of Total Sales", format="%.1f%%"),
                        },
                        use_container_width=True,
                    )

                    # Summary Metric Highlights
                    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
                    class_a_items = summary_df.loc[summary_df["Final Inventory Class"] == "A", "Number of Items"].values[0]
                    class_a_sales = summary_df.loc[summary_df["Final Inventory Class"] == "A", "Total Sales ($)"].values[0]
                    class_b_items = summary_df.loc[summary_df["Final Inventory Class"] == "B", "Number of Items"].values[0]
                    class_b_sales = summary_df.loc[summary_df["Final Inventory Class"] == "B", "Total Sales ($)"].values[0]

                    c_col1.metric("Class A Items", f"{class_a_items:,}")
                    c_col2.metric("Class A Total Sales", f"${class_a_sales:,.2f}")
                    c_col3.metric("Class B Items", f"{class_b_items:,}")
                    c_col4.metric("Class B Total Sales", f"${class_b_sales:,.2f}")

                    # Full Updated Dataframe with all 6 new calculated columns
                    st.subheader("📑 Full Classified Inventory Dataframe")
                    st.dataframe(
                        res_class["classified_df"],
                        column_config={
                            "Sales ($)": st.column_config.NumberColumn(format="$%.2f"),
                            "Hits": st.column_config.NumberColumn(format="%d"),
                            "Sales Score": st.column_config.NumberColumn(format="%d"),
                            "Hits Score": st.column_config.NumberColumn(format="%d"),
                            "Combined Weighted Score": st.column_config.NumberColumn(format="%.2f"),
                        },
                        use_container_width=True,
                    )

            # ---------------------------------------------------------------
            # 2. Existing Inventory Units & Value Analysis
            # ---------------------------------------------------------------
            if is_units_analysis_dataset:
                res = analyze_inventory_csv(raw_df)

                if not res["valid"]:
                    st.error(f"❌ Missing required analysis columns: {', '.join(res['missing_columns'])}")
                    st.info("The inventory analysis dataset requires: **Product**, **Category**, **Units**, and **Unit Cost**.")
                    if not is_classification_dataset:
                        st.subheader("📄 Uploaded Dataframe")
                        st.dataframe(raw_df, use_container_width=True)
                else:
                    st.divider()
                    st.subheader("📈 Inventory Value & Units Summary")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Total Units", f"{res['total_units']:,.0f}")
                    k2.metric("Average Unit Cost", f"${res['average_unit_cost']:,.2f}")
                    k3.metric("Product with Highest Units", f"{res['top_product']} ({res['top_product_units']:,.0f} units)")
                    k4.metric("Total Inventory Value", f"${res['total_inventory_value']:,.2f}")

                    # Total Units by Category
                    st.subheader("📦 Total Units by Category")
                    st.dataframe(
                        res["category_summary"],
                        column_config={
                            "Total Units": st.column_config.NumberColumn(format="%d"),
                        },
                        use_container_width=True,
                    )

                    # Enriched dataframe with Inventory Value column
                    st.subheader("📑 Enriched Inventory Data (with Inventory Value)")
                    st.dataframe(
                        res["df_with_inventory_value"],
                        column_config={
                            "Unit Cost": st.column_config.NumberColumn(format="$%.2f"),
                            "Inventory Value": st.column_config.NumberColumn(format="$%.2f"),
                            "Units": st.column_config.NumberColumn(format="%d"),
                        },
                        use_container_width=True,
                    )

            # If neither format was recognized
            if not is_classification_dataset and not is_units_analysis_dataset:
                st.warning("⚠️ Unrecognized CSV schema. Please upload a file containing either:")
                st.markdown("- **Inventory Classification:** `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`")
                st.markdown("- **Inventory Analysis:** `Product`, `Category`, `Units`, `Unit Cost`")
                st.subheader("📄 Uploaded Dataframe")
                st.dataframe(raw_df, use_container_width=True)


# ===========================================================================
# Main Execution Dispatcher
# ===========================================================================
def is_running_in_streamlit() -> bool:
    """Reliable detection for Streamlit Cloud and local `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        # Fallback checks
        return (
            "streamlit" in os.path.basename(sys.argv[0]).lower()
            or os.environ.get("STREAMLIT_SERVER_PORT") is not None
            or (len(sys.argv) > 1 and sys.argv[1] == "run")
        )


def main():
    if is_running_in_streamlit():
        run_streamlit_app()
        return

    # Otherwise run CLI / Terminal Runner
    parser = argparse.ArgumentParser(
        description="Supply Chain Analytics Inventory Decision Agent CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Natural-language inventory question to solve directly.",
    )
    parser.add_argument(
        "-b", "--benchmark",
        action="store_true",
        help="Run standard 5-model supply chain benchmark tests.",
    )

    args = parser.parse_args()
    agent = InventoryAgent()

    from rich.console import Console
    console = Console()

    if args.question:
        run_cli_single(agent, args.question, console)
    elif args.benchmark:
        console.print("\n[bold yellow]Executing Supply Chain Analytics Benchmark Suite...[/bold yellow]\n")
        for item in BENCHMARK_SCENARIOS:
            console.rule(f"[bold blue]{item['title']}[/bold blue]")
            run_cli_single(agent, item["question"], console)
            console.print()
    else:
        run_cli_interactive(agent)


if __name__ == "__main__":
    main()
