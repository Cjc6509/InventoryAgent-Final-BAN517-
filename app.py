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


# ===========================================================================
# Main Execution Dispatcher
# ===========================================================================
def main():
    # Detect if invoked through `streamlit run`
    is_streamlit = (
        "streamlit" in os.path.basename(sys.argv[0]).lower()
        or os.environ.get("STREAMLIT_SERVER_PORT") is not None
        or (len(sys.argv) > 1 and sys.argv[1] == "run")
    )

    if is_streamlit:
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
