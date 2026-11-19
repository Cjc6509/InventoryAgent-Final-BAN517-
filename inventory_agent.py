"""
Supply Chain Analytics - Inventory Decision Agent
Coordinates the complete decision-making pipeline:
Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain

Features advanced natural-language extraction recognizing industry terminology,
synonyms, multi-word attributes, bracket schedules, and contextual inventory parameters.
"""

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import inventory_tools as it


class InventoryModelType:
    EOQ = "EOQ"
    QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    NEWSVENDOR = "NEWSVENDOR"
    UNKNOWN = "UNKNOWN"


@dataclass
class AgentResponse:
    question: str
    model_type: str
    model_name: str
    raw_inputs: Dict[str, Any]
    calculation_result: Any
    verification_passed: bool
    verification_notes: List[str]
    explanation: str
    error: Optional[str] = None
    structured_metrics: Dict[str, Any] = field(default_factory=dict)


class InventoryAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.
    Accepts natural-language inventory questions and supports 5 core models:
    1. Economic Order Quantity (EOQ)
    2. Quantity Discount
    3. Safety Stock & Reorder Point
    4. Fixed-Period Review
    5. Newsvendor
    """

    MODEL_DISPLAY_NAMES = {
        InventoryModelType.EOQ: "Economic Order Quantity (EOQ)",
        InventoryModelType.QUANTITY_DISCOUNT: "Quantity Discount Analysis",
        InventoryModelType.SAFETY_STOCK_ROP: "Safety Stock & Reorder Point (Continuous Review)",
        InventoryModelType.FIXED_PERIOD_REVIEW: "Fixed-Period Review (Order-Up-To Periodic Review)",
        InventoryModelType.NEWSVENDOR: "Newsvendor Single-Period Optimization",
    }

    def __init__(self):
        pass

    # -----------------------------------------------------------------------
    # Step 1: Identify Model
    # -----------------------------------------------------------------------
    def identify_model(self, question: str) -> str:
        """
        Classify customer natural-language question into one of the 5 inventory models.
        Examines the entire question context before classifying.
        """
        text = question.lower()

        # 1. Quantity Discount
        if any(term in text for term in [
            "discount", "price break", "volume discount", "bulk discount", "tiers", "bracket",
            "brackets", "all-units", "quantity break", "price tier", "volume tier", "price schedule",
            "pricing tier", "pricing tiers", "price breaks", "quantity breaks",
        ]) or (
            re.search(r"\b[0-9,]+\s*(?:to|-)\s*[0-9,]+\s*units?", text) and re.search(r"\b[0-9,]+\s*\+\s*units?", text)
        ) or (
            "price is" in text and ("units" in text and ("+" in text or "-" in text or "to" in text))
        ):
            return InventoryModelType.QUANTITY_DISCOUNT

        # 2. Fixed-Period Review
        if any(term in text for term in [
            "fixed period", "periodic review", "order-up-to", "order up to", "target level",
            "target inventory", "review period", "review interval", "protection interval",
            "protection period", "periodic replenishment", "every 2 weeks", "every week",
            "every month", "weekly review", "monthly review", "inventory position"
        ]) and not any(term in text for term in ["newsvendor", "single period", "perishable"]):
            return InventoryModelType.FIXED_PERIOD_REVIEW

        # 3. Newsvendor (Single-Period / Perishable / Seasonal)
        if any(term in text for term in [
            "newsvendor", "newsboy", "single period", "single-period", "perishable", "seasonal",
            "salvage", "underage", "overage", "critical ratio", "critical fractile", "christmas tree",
            "fashion item", "unsold units", "leftover value", "markdown price", "shortage cost",
            "excess cost", "spoilage"
        ]):
            return InventoryModelType.NEWSVENDOR

        # 4. Safety Stock & Reorder Point (Continuous Review)
        if any(term in text for term in [
            "safety stock", "reorder point", "rop", "service level", "stockout risk",
            "uncertain lead time", "lead time demand", "continuous review", "std dev",
            "standard deviation", "lead-time standard deviation", "lead time variability",
            "cycle service level"
        ]) and not any(term in text for term in ["eoq", "economic order"]):
            return InventoryModelType.SAFETY_STOCK_ROP

        # If question asks for ROP specifically alongside standard EOQ, check if stochastic
        if ("reorder point" in text or "safety stock" in text) and (
            "service level" in text or "standard deviation" in text or "variance" in text
        ):
            return InventoryModelType.SAFETY_STOCK_ROP

        # 5. Economic Order Quantity (EOQ)
        if any(term in text for term in [
            "eoq", "economic order quantity", "annual demand", "setup cost", "setup expense",
            "ordering cost", "order cost", "cost per order", "cost of placing an order",
            "ordering/setup cost", "holding cost", "carrying cost", "order quantity",
            "optimal batch size", "how many units should we order", "how much should we order",
            "batch size"
        ]):
            return InventoryModelType.EOQ

        # Default heuristic fallbacks based on prominent terms
        if "lead time" in text and "reorder" in text:
            return InventoryModelType.SAFETY_STOCK_ROP
        if "demand" in text and ("order" in text or "cost" in text):
            return InventoryModelType.EOQ

        return InventoryModelType.UNKNOWN

    # -----------------------------------------------------------------------
    # Step 2: Extract Inputs with Extensive Synonym & Terminology Support
    # -----------------------------------------------------------------------
    def extract_inputs(
        self, question: str, model_type: str, custom_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract numerical and structural parameters from natural-language text.
        Examines the entire question thoroughly before determining missing parameters.
        """
        inputs: Dict[str, Any] = {}
        text = question

        # Helper: Clean string to float
        def parse_num(val_str: str) -> Optional[float]:
            cleaned = val_str.replace(",", "").replace("$", "").rstrip("., ")
            try:
                return float(cleaned)
            except ValueError:
                return None

        # Helper: Search number after patterns
        def find_after(patterns: List[str]) -> Optional[float]:
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    val = parse_num(m.group(1))
                    if val is not None:
                        return val
            return None

        # Helper: Search number before patterns
        def find_before(patterns: List[str]) -> Optional[float]:
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    val = parse_num(m.group(1))
                    if val is not None:
                        return val
            return None

        # -------------------------------------------------------------------
        # A. Ordering Cost Synonyms (EOQ & Quantity Discount)
        # -------------------------------------------------------------------
        # Recognizes: ordering cost, order cost, setup cost, setup expense,
        # cost per order, cost of placing an order, ordering/setup cost,
        # fixed cost per order, order setup fee, order fee, expense per order
        order_cost_patterns = [
            # Explicit keywords followed by number
            r"(?:ordering\s*\/\s*setup\s*cost|setup\s*\/\s*ordering\s*cost|ordering\s*and\s*setup\s*cost)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"(?:cost\s+of\s+placing\s+an\s+order|cost\s+to\s+place\s+an\s+order|cost\s+to\s+order)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"(?:cost\s+per\s+order|expense\s+per\s+order|fee\s+per\s+order)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"(?:setup\s+expense|setup\s+cost|fixed\s+setup\s+cost|order\s+setup\s+fee)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"(?:ordering\s+cost|order\s+cost|fixed\s+ordering\s+cost|order\s+placement\s+cost)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            # Number preceding keywords
            r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+order|to\s+place\s+an\s+order|setup\s+fee|setup\s+expense|ordering\s+fee)",
            r"([0-9]+(?:\.[0-9]+)?)\s*dollars?\s*(?:per\s+order|to\s+place\s+an\s+order|setup\s+expense|ordering\s+cost)",
        ]
        order_cost = find_after(order_cost_patterns)

        # -------------------------------------------------------------------
        # B. Demand Synonyms
        # -------------------------------------------------------------------
        # Distinguishes annual demand vs daily demand vs general average demand
        annual_demand_patterns = [
            r"(?:annual\s+demand|yearly\s+demand|annual\s+consumption|annual\s+sales|annual\s+requirement)\s*(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+per\s+year|units\/year|annual\s+units)",
            r"sells\s*([0-9,]+(?:\.[0-9]+)?)\s*units\s*(?:per\s+year|annually)",
            r"(?:demand\s+per\s+year|sales\s+per\s+year)\s*(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
        ]
        daily_demand_patterns = [
            r"(?:average\s+daily\s+demand|mean\s+daily\s+demand|daily\s+demand|demand\s+per\s+day)\s*(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*(?:units\s+per\s+day|units\/day|daily\s+units)",
            r"sells\s+(?:an\s+average\s+of\s+)?([0-9,]+(?:\.[0-9]+)?)\s*units\s+per\s+day",
        ]
        general_demand_patterns = [
            r"(?:average\s+demand|mean\s+demand|expected\s+demand|demand\s+mean|demand\s+rate|forecasted\s+demand)\s*(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s+is\s+(?:normally\s+distributed|normal)?\s*(?:with\s+)?mean\s*(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"(?:with\s+)?mean\s*(?:demand\s+)?(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            r"demand\s*(?:is|of|:|=)\s*([0-9,]+(?:\.[0-9]+)?)",
            r"([0-9,]+(?:\.[0-9]+)?)\s*units\s+of\s+demand",
            r"sells\s+(?:approximately\s+|an\s+average\s+of\s+)?([0-9,]+(?:\.[0-9]+)?)\s*units",
            r"([0-9,]+(?:\.[0-9]+)?)\s*units",
        ]

        annual_demand = find_after(annual_demand_patterns)
        daily_demand = find_after(daily_demand_patterns)
        general_demand = find_after(general_demand_patterns)
        resolved_demand = annual_demand or daily_demand or general_demand

        # -------------------------------------------------------------------
        # C. Holding Cost: Fixed Dollar Amount vs Percentage Rate
        # -------------------------------------------------------------------
        # Percentage holding cost rate (e.g. 20%, 0.20)
        holding_pct_match = re.search(
            r"(?:percentage\s+holding\s+cost|holding\s+cost\s+rate|carrying\s+rate|holding\s+rate|annual\s+carrying\s+rate|carrying\s+cost\s+of|holding\s+cost\s+of)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*%",
            text, re.IGNORECASE
        )
        if not holding_pct_match:
            holding_pct_match = re.search(
                r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:percentage\s+holding\s+cost|holding\s+cost|carrying\s+cost|annual\s+carrying|holding\s+rate|carrying\s+rate|of\s+unit\s+price|of\s+purchase\s+price|of\s+unit\s+cost)",
                text, re.IGNORECASE
            )

        holding_rate = None
        if holding_pct_match:
            try:
                val = float(holding_pct_match.group(1))
                holding_rate = val / 100.0 if val > 1.0 else val
            except ValueError:
                holding_rate = None

        # Fixed dollar holding cost ($/unit/year)
        fixed_holding_patterns = [
            r"(?:fixed\s+holding\s+cost|holding\s+cost\s+per\s+unit|carrying\s+cost\s+per\s+unit|storage\s+cost|holding\s+expense|cost\s+to\s+carry|cost\s+of\s+carrying|holding\s+fee)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)",
            r"(?:holding\s+cost|carrying\s+cost)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)",
            r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+year|\/unit\/year|per\s+unit\/year|per\s+year\s+to\s+hold|holding\s+fee|carrying\s+charge)",
        ]
        holding_cost = find_after(fixed_holding_patterns)

        # -------------------------------------------------------------------
        # D. Unit Purchase Price / Cost Synonyms
        # -------------------------------------------------------------------
        unit_price_patterns = [
            r"(?:unit\s+purchase\s+price|unit\s+purchase\s+cost|purchase\s+price|purchase\s+cost|wholesale\s+price|wholesale\s+cost|acquisition\s+cost)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"(?:unit\s+price|unit\s+cost|item\s+cost|cost\s+per\s+unit|price\s+per\s+unit)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|each|to\s+buy|to\s+purchase|wholesale)",
            r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:each|per\s+unit)(?!\s*(?:per\s+year|holding|carrying|setup|order))",
        ]
        unit_cost = find_after(unit_price_patterns)
        # Avoid collision if unit_cost accidentally extracted order_cost or holding_cost
        if unit_cost is not None and unit_cost in (order_cost, holding_cost):
            unit_cost = None

        # -------------------------------------------------------------------
        # E. Lead Time & Lead Time Standard Deviation
        # -------------------------------------------------------------------
        # Lead time standard deviation (check FIRST so standard deviation is not stolen by mean lead time)
        lt_std_patterns = [
            r"(?:lead-time\s+standard\s+deviation|lead\s+time\s+standard\s+deviation|standard\s+deviation\s+of\s+lead\s+time|lead\s+time\s+std\s+dev|std\s+dev\s+of\s+lead\s+time|lead-time\s+std\s+dev|sigma_l|sigma\s+l|lead\s+time\s+variability)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:lead\s+time\s+varies\s+(?:with\s+|by\s+)?(?:a\s+)?standard\s+deviation\s+of)\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:standard\s+deviation|std\s+dev)[^0-9]*(?:of|is|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:lead\s+time)?",
        ]
        lead_time_std = find_after(lt_std_patterns)

        # Average / Mean lead time
        lead_time_patterns = [
            r"(?:average\s+lead\s+time|mean\s+lead\s+time|expected\s+lead\s+time|supplier\s+lead\s+time|replenishment\s+lead\s+time|delivery\s+lead\s+time|lead\s+time)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:days\s+lead\s+time|day\s+lead\s+time|weeks\s+lead\s+time|week\s+lead\s+time)",
            r"takes\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?)\s*(?:to\s+arrive|lead\s+time)?",
        ]
        lead_time = find_after(lead_time_patterns)

        # -------------------------------------------------------------------
        # F. Demand Standard Deviation
        # -------------------------------------------------------------------
        demand_std_patterns = [
            r"(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|demand\s+std\s+dev|std\s+dev\s+of\s+demand|daily\s+demand\s+standard\s+deviation|daily\s+demand\s+std\s+dev|standard\s+deviation\s+in\s+demand|sigma_d|sigma\s+d|demand\s+variability)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)",
            r"(?:standard\s+deviation|std\s+dev)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|units\/day|units\s+per\s+day)?",
            r"([0-9]+(?:\.[0-9]+)?)\s*(?:units\s+standard\s+deviation|units\s+std\s+dev)",
            r"demand\s+varies\s+(?:with\s+|by\s+)?(?:a\s+)?standard\s+deviation\s+of\s*([0-9]+(?:\.[0-9]+)?)",
        ]
        demand_std = find_after(demand_std_patterns)

        # -------------------------------------------------------------------
        # G. Service Level / Z-Score
        # -------------------------------------------------------------------
        sl_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+)?service\s+level", text, re.IGNORECASE
        )
        if not sl_match:
            sl_match = re.search(
                r"(?:cycle\s+service\s+level|target\s+service\s+level|customer\s+service\s+level|service\s+level|csl|fill\s+probability|in-stock\s+probability)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*%?",
                text, re.IGNORECASE
            )
        service_level = None
        if sl_match:
            try:
                sl_val = float(sl_match.group(1).rstrip("., "))
                service_level = sl_val / 100.0 if sl_val > 1.0 else sl_val
            except ValueError:
                service_level = None

        z_match = re.search(r"(?:z-score|z\s+value|safety\s+factor\s+z)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        z_score = parse_num(z_match.group(1)) if z_match else None

        # Direct safety stock if explicitly provided
        direct_ss = find_after([
            r"(?:safety\s+stock|buffer\s+stock)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*units?",
            r"([0-9]+(?:\.[0-9]+)?)\s*units\s+of\s+safety\s+stock",
        ])

        # -------------------------------------------------------------------
        # MODEL SPECIFIC POPULATION & EXTRACTIONS
        # -------------------------------------------------------------------
        if model_type == InventoryModelType.EOQ:
            inputs["demand"] = annual_demand or resolved_demand
            inputs["order_cost"] = order_cost
            inputs["holding_cost"] = holding_cost
            inputs["unit_cost"] = unit_cost
            inputs["holding_rate"] = holding_rate
            inputs["lead_time_days"] = lead_time

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            inputs["demand"] = annual_demand or resolved_demand
            inputs["order_cost"] = order_cost
            inputs["holding_rate"] = holding_rate
            inputs["fixed_holding_cost"] = holding_cost

            # Comprehensive Price Break Parser:
            # Handles:
            # 1. "1 to 499 at $5.00" / "1-499: $5.00"
            # 2. "under 500: $5.00" / "less than 500: $5.00" / "< 500 at $5.00"
            # 3. "500 to 999: $4.50"
            # 4. "1000 or more at $3.90" / "1000+ at $3.90" / ">= 1000: $3.90"
            # 5. Markdown table rows: "| 1-499 | $5.00 |"
            brackets: List[Dict[str, Any]] = []

            # Lower tier bounds like "under 500 units at $5.00" or "less than 500 at $5.00"
            lower_bracket_match = re.search(
                r"(?:under|less\s+than|<)\s*([0-9,]+)\b\s*(?:units)?(?:\s+(?:at|@|:|=|are|for))?\s*(?:unit\s+price|price)?\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                text, re.IGNORECASE
            )
            if lower_bracket_match:
                up_to = float(lower_bracket_match.group(1).replace(",", ""))
                price = float(lower_bracket_match.group(2))
                brackets.append({"min_qty": 1.0, "max_qty": up_to - 1.0, "price": price})

            # Inverted phrasing: "$10 for 1-499 units", "$8 for 500+ units"
            inverted_ranges = re.findall(
                r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|each)?\s*(?:for|if|when)\s*([0-9,]+)\s*(?:to|-)\s*([0-9,]+)\b\s*(?:units)?",
                text, re.IGNORECASE
            )
            for inv in inverted_ranges:
                brackets.append({
                    "min_qty": float(inv[1].replace(",", "")),
                    "max_qty": float(inv[2].replace(",", "")),
                    "price": float(inv[0]),
                })

            inverted_top = re.findall(
                r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|each)?\s*(?:for|if|when)\s*([0-9,]+)\s*(?:\+|or\s+more|and\s+above|units\s+or\s+more)\s*(?:units)?",
                text, re.IGNORECASE
            )
            for inv_t in inverted_top:
                brackets.append({
                    "min_qty": float(inv_t[1].replace(",", "")),
                    "max_qty": None,
                    "price": float(inv_t[0]),
                })

            # Range matches: "1 to 499 at $5.00", "500-999: $4.50"
            bracket_matches = re.findall(
                r"\b([0-9,]+)\s*(?:to|-)\s*([0-9,]+)\b\s*(?:units)?(?:\s+(?:at|@|:|=|are|for)|,\s*(?:at|for|price))?\s*(?:unit\s+price|price)?\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                text, re.IGNORECASE
            )
            for b in bracket_matches:
                brackets.append({
                    "min_qty": float(b[0].replace(",", "")),
                    "max_qty": float(b[1].replace(",", "")),
                    "price": float(b[2]),
                })

            # Unbounded top tier: "1000+ at $3.90", "1000 or more at $3.90", ">= 1000: $3.90"
            top_match = re.search(
                r"\b([0-9,]+)\s*(?:\+|or\s+more|and\s+above|units\s+or\s+more|>=\s*|at\s+least\s*)\s*(?:units)?(?:\s+(?:at|@|:|=|are|for))?\s*(?:unit\s+price|price)?\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                text, re.IGNORECASE
            )
            if not top_match:
                top_match = re.search(
                    r"(?:>=\s*|at\s+least\s*)\s*([0-9,]+)\b\s*(?:units)?(?:\s+(?:at|@|:|=|are|for))?\s*(?:unit\s+price|price)?\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                    text, re.IGNORECASE
                )
            if top_match:
                brackets.append({
                    "min_qty": float(top_match.group(1).replace(",", "")),
                    "max_qty": None,
                    "price": float(top_match.group(2)),
                })

            # Remove any duplicates and sort
            unique_brackets = []
            seen_mins = set()
            for b in sorted(brackets, key=lambda x: x["min_qty"]):
                if b["min_qty"] not in seen_mins:
                    seen_mins.add(b["min_qty"])
                    unique_brackets.append(b)

            if unique_brackets:
                inputs["price_brackets"] = unique_brackets

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            inputs["daily_demand_mean"] = daily_demand or resolved_demand
            inputs["lead_time_mean"] = lead_time
            inputs["daily_demand_std"] = demand_std or 0.0
            inputs["lead_time_std"] = lead_time_std or 0.0
            inputs["service_level"] = service_level
            inputs["z_score"] = z_score
            inputs["unit_holding_cost"] = holding_cost
            if direct_ss is not None:
                inputs["safety_stock_fixed"] = direct_ss

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            # Review period extraction
            review_period = find_after([
                r"(?:review\s+period|review\s+interval|review\s+cycle|reviewed\s+every|order\s+interval|review\s+frequency)\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days?|weeks?|months?)?",
                r"every\s*([0-9]+(?:\.[0-9]+)?)\s*(?:days|day)",
            ])
            if review_period is None:
                if re.search(r"every\s+(?:2|two)\s+weeks", text, re.IGNORECASE):
                    review_period = 14.0
                elif re.search(r"every\s+(?:1|one)?\s*week", text, re.IGNORECASE):
                    review_period = 7.0
                elif re.search(r"every\s+(?:1|one)?\s*month", text, re.IGNORECASE):
                    review_period = 30.0

            # Inventory position direct vs component breakdown
            inv_position = find_after([
                r"(?:inventory\s+position|net\s+stock|current\s+inventory\s+position)\s*(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ])
            on_hand = find_after([
                r"(?:on-hand\s+inventory|on\s+hand\s+inventory|on\s+hand|current\s+inventory|stock\s+on\s+hand|physically\s+available)\s*(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ]) or 0.0
            on_order = find_after([
                r"(?:on-order\s+inventory|on\s+order\s+inventory|on\s+order|in\s+transit|pipeline\s+inventory|scheduled\s+receipts|open\s+orders)\s*(?:is|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ]) or 0.0
            backorders = find_after([
                r"(?:backorders|backlogged\s+orders|backlogged|backlog|unfilled\s+orders)\s*(?:is|are|of|:|=)?\s*([0-9,]+(?:\.[0-9]+)?)",
            ]) or 0.0

            inputs["review_period"] = review_period
            inputs["lead_time"] = lead_time
            inputs["daily_demand_mean"] = daily_demand or resolved_demand
            inputs["daily_demand_std"] = demand_std or 0.0
            inputs["service_level"] = service_level
            inputs["z_score"] = z_score
            inputs["on_hand_inventory"] = on_hand
            inputs["on_order_inventory"] = on_order
            inputs["backorders"] = backorders
            if direct_ss is not None:
                inputs["safety_stock_fixed"] = direct_ss
            if inv_position is not None:
                inputs["inventory_position_fixed"] = inv_position

        elif model_type == InventoryModelType.NEWSVENDOR:
            # Selling price synonyms
            selling_price = find_after([
                r"(?:unit\s+selling\s+price|selling\s+price|retail\s+price|sale\s+price|revenue\s+per\s+unit|price\s+per\s+unit)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"sells?\s+(?:for|at)\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"sells?[^0-9$]*\$([0-9]+(?:\.[0-9]+)?)",
                r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:retail|selling\s+price|sale\s+price)",
            ])
            # Purchase cost synonyms
            purchase_cost = find_after([
                r"(?:wholesale\s+(?:purchase\s+)?cost|wholesale\s+price|wholesale\s+cost|unit\s+purchase\s+cost|purchase\s+cost|unit\s+cost|cost\s+per\s+unit|acquisition\s+cost|cost\s+of\s+goods|cost\s+to\s+buy|cost\s+to\s+purchase|costs?)\s*(?:is|are|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit|to\s+buy|to\s+purchase|wholesale|to\s+produce)?",
                r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:cost|wholesale|acquisition)",
            ])
            # Salvage value synonyms
            salvage_val = find_after([
                r"(?:salvaged?\s+(?:for|at)|salvage\s+value|salvage\s+price|salvage\s+of|liquidation\s+value|scrap\s+value|clearance\s+price|markdown\s+price|residual\s+value|scrapped\s+for|discounted\s+to)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
                r"\$([0-9]+(?:\.[0-9]+)?)\s*(?:salvage|scrap|liquidation|markdown)",
            ]) or 0.0

            # Direct Underage (Cu) and Overage (Co) costs
            cu_direct = find_after([
                r"(?:underage\s+cost|shortage\s+cost|cost\s+of\s+underage|cost\s+of\s+shortage|underage\s+expense|c_u|cu)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            ])
            co_direct = find_after([
                r"(?:overage\s+cost|excess\s+cost|cost\s+of\s+overage|cost\s+of\s+excess|overage\s+expense|c_o|co)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            ])

            penalty = find_after([
                r"(?:shortage\s+penalty|penalty\s+cost|penalty|goodwill\s+loss|stockout\s+penalty)\s*(?:is|of|:|=)?\s*\$?([0-9]+(?:\.[0-9]+)?)",
            ]) or 0.0

            inputs["selling_price"] = selling_price
            inputs["purchase_cost"] = purchase_cost
            inputs["salvage_value"] = salvage_val
            inputs["penalty_cost"] = penalty
            inputs["demand_mean"] = resolved_demand
            inputs["demand_std"] = demand_std or 0.0
            if cu_direct is not None:
                inputs["underage_cost_direct"] = cu_direct
            if co_direct is not None:
                inputs["overage_cost_direct"] = co_direct

        # Apply custom overrides if provided
        if custom_overrides:
            inputs.update(custom_overrides)

        return inputs

    # -----------------------------------------------------------------------
    # Step 3: Select Python Tool
    # -----------------------------------------------------------------------
    def select_tool(self, model_type: str) -> Tuple[Callable, str]:
        """
        Map model type to the corresponding Python mathematical engine in inventory_tools.
        """
        tool_map = {
            InventoryModelType.EOQ: (it.calculate_eoq, "inventory_tools.calculate_eoq"),
            InventoryModelType.QUANTITY_DISCOUNT: (it.calculate_quantity_discount, "inventory_tools.calculate_quantity_discount"),
            InventoryModelType.SAFETY_STOCK_ROP: (it.calculate_rop, "inventory_tools.calculate_rop"),
            InventoryModelType.FIXED_PERIOD_REVIEW: (it.calculate_periodic_review, "inventory_tools.calculate_periodic_review"),
            InventoryModelType.NEWSVENDOR: (it.calculate_newsvendor, "inventory_tools.calculate_newsvendor"),
        }
        if model_type not in tool_map:
            raise ValueError(f"No execution tool available for model type '{model_type}'")
        return tool_map[model_type]

    # -----------------------------------------------------------------------
    # Step 4: Calculate
    # -----------------------------------------------------------------------
    def calculate(self, tool_fn: Callable, inputs: Dict[str, Any]) -> Any:
        """
        Execute pure Python mathematical calculations with validated parameters.
        Filter out None values to allow defaults in tool functions.
        """
        clean_kwargs = {k: v for k, v in inputs.items() if v is not None}
        return tool_fn(**clean_kwargs)

    # -----------------------------------------------------------------------
    # Step 5: Verify
    # -----------------------------------------------------------------------
    def verify(self, model_type: str, result: Any) -> Tuple[bool, List[str]]:
        """
        Verify results against mathematical consistency criteria and invariants.
        """
        if hasattr(result, "verification_passed") and hasattr(result, "verification_notes"):
            return result.verification_passed, list(result.verification_notes)
        return True, ["Calculations completed successfully."]

    # -----------------------------------------------------------------------
    # Step 6: Explain
    # -----------------------------------------------------------------------
    def explain(self, model_type: str, inputs: Dict[str, Any], result: Any, verification_notes: List[str]) -> str:
        """
        Generate educational, step-by-step verification explanation containing all 8 required sections:
        1. Inputs Extracted
        2. Variable Definitions
        3. Formula
        4. Formula with Substituted Values
        5. Intermediate Calculations
        6. Final Answer
        7. Verification Checks
        8. Assumptions
        All numerical calculations originate strictly from the deterministic Python tool.
        """
        lines = []
        title = self.MODEL_DISPLAY_NAMES.get(model_type, "Inventory Optimization")
        lines.append(f"## 📦 {title} — Step-by-Step Student Verification")
        lines.append("")

        # 1. Inputs Extracted
        lines.append("### 1. Inputs Extracted")
        if hasattr(result, "inputs_extracted") and isinstance(result.inputs_extracted, dict):
            # Display primary mathematical parameters
            for k, v in result.inputs_extracted.items():
                # Avoid printing duplicate alias keys in the pedagogical view
                if k in ("demand", "order_cost", "holding_cost", "unit_cost", "holding_rate",
                        "lead_time_days", "daily_demand_mean", "lead_time_mean", "daily_demand_std",
                        "lead_time_std", "service_level", "review_period", "lead_time",
                        "selling_price", "purchase_cost", "salvage_value", "demand_mean", "demand_std"):
                    continue
                if v is None:
                    continue
                if isinstance(v, list) and k == "price_brackets":
                    bracket_strs = []
                    for b in v:
                        max_q = b.get("max_qty")
                        max_str = f"{max_q:,.0f}" if max_q is not None else "∞"
                        min_q = b.get("min_qty", 0)
                        pr = b.get("price", 0)
                        bracket_strs.append(f"[{min_q:,.0f} - {max_str}: ${pr:.2f}]")
                    lines.append(f"- **{k}**: {', '.join(bracket_strs)}")
                elif isinstance(v, float):
                    if abs(v) < 0.01 and v != 0:
                        lines.append(f"- **{k}**: `{v:,.6f}`")
                    elif abs(v) < 1.0 and v != 0:
                        lines.append(f"- **{k}**: `{v:,.4f}`")
                    else:
                        lines.append(f"- **{k}**: `{v:,.2f}`")
                else:
                    lines.append(f"- **{k}**: `{v}`")
        lines.append("")

        # 2. Variable Definitions
        lines.append("### 2. Variable Definitions")
        if hasattr(result, "variable_definitions") and isinstance(result.variable_definitions, dict):
            for var_sym, var_def in result.variable_definitions.items():
                lines.append(f"- **`{var_sym}`**: {var_def}")
        lines.append("")

        # 3. Formula
        lines.append("### 3. Formula")
        if hasattr(result, "formula") and isinstance(result.formula, dict):
            for f_name, f_val in result.formula.items():
                clean_name = f_name.replace("_", " ").title()
                lines.append(f"- **{clean_name}:** `{f_val}`")
        lines.append("")

        # 4. Formula with Substituted Values
        lines.append("### 4. Formula with Substituted Values")
        if hasattr(result, "formula_with_substituted_values") and isinstance(result.formula_with_substituted_values, dict):
            for s_name, s_val in result.formula_with_substituted_values.items():
                clean_name = s_name.replace("_", " ").title()
                lines.append(f"- **{clean_name}:** {s_val}")
        lines.append("")

        # 5. Intermediate Calculations
        lines.append("### 5. Intermediate Calculations")
        if hasattr(result, "intermediate_calculations") and isinstance(result.intermediate_calculations, dict):
            for i_name, i_val in result.intermediate_calculations.items():
                if isinstance(i_val, list):
                    continue  # Table rendered below
                clean_name = i_name.replace("_", " ").title()
                if isinstance(i_val, float):
                    lines.append(f"- **{clean_name}:** `{i_val:,.4f}`" if (abs(i_val) < 1 and abs(i_val) > 0) else f"- **{clean_name}:** `{i_val:,.2f}`")
                else:
                    lines.append(f"- **{clean_name}:** `{i_val}`")

        # Specific intermediate breakdown tables
        if model_type == InventoryModelType.QUANTITY_DISCOUNT and hasattr(result, "tier_evaluations"):
            lines.append("")
            lines.append("**Discount Schedule Evaluation Table:**")
            lines.append("| Tier | Quantity Range | Unit Price | Unconstrained EOQ | Candidate Q | Purchase Cost | Holding Cost | Ordering Cost | Total Cost | Feasibility Status |")
            lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
            for t in result.tier_evaluations:
                max_str = f"{t.max_quantity:,.0f}" if t.max_quantity is not None else "∞"
                is_opt = " **(OPTIMAL)**" if t.tier_index == result.optimal_tier_index else ""
                lines.append(
                    f"| {t.tier_index} | [{t.min_quantity:,.0f}, {max_str}) | ${t.unit_price:.2f} | {t.unconstrained_eoq:,.1f} | "
                    f"**{t.candidate_order_quantity:,.0f}** | ${t.annual_purchase_cost:,.2f} | ${t.annual_holding_cost:,.2f} | "
                    f"${t.annual_ordering_cost:,.2f} | **${t.total_acquisition_cost:,.2f}**{is_opt} | {t.feasibility_status} |"
                )
        lines.append("")

        # 6. Final Answer
        lines.append("### 6. Final Answer")
        if hasattr(result, "final_answer") and isinstance(result.final_answer, dict):
            for a_name, a_val in result.final_answer.items():
                clean_name = a_name.replace("_", " ").title()
                if isinstance(a_val, float):
                    lines.append(f"- **{clean_name}:** `{a_val:,.2f}`")
                else:
                    lines.append(f"- **{clean_name}:** `{a_val}`")
        lines.append("")

        # 7. Verification Checks
        lines.append("### 7. Verification Checks")
        if hasattr(result, "verification_checks") and isinstance(result.verification_checks, dict):
            audit_log = result.verification_checks.get("audit_log", [])
            for note in audit_log:
                lines.append(f"- ✅ {note}")
            for v_name, v_check in result.verification_checks.items():
                if isinstance(v_check, dict) and "passed" in v_check:
                    p_sym = "✅" if v_check["passed"] else "❌"
                    cond = v_check.get("condition", "")
                    clean_check_name = v_name.replace("_", " ").title()
                    lines.append(f"- {p_sym} **{clean_check_name}:** {cond}")
            all_passed = result.verification_checks.get("all_checks_passed", True)
            lines.append(f"- **Overall Verification Status:** `{'PASSED' if all_passed else 'FAILED'}`")
        lines.append("")

        # 8. Assumptions
        lines.append("### 8. Assumptions")
        if hasattr(result, "assumptions") and isinstance(result.assumptions, list):
            for idx, asm in enumerate(result.assumptions, start=1):
                lines.append(f"{idx}. {asm}")
        lines.append("")

        return "\n".join(lines)

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Main Agent Pipeline Entry Point
    # -----------------------------------------------------------------------
    def process_question(self, question: str, custom_inputs: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute full end-to-end pipeline:
        Customer Question -> Identify Model -> Extract Inputs -> Select Python Tool -> Calculate -> Verify -> Explain
        """
        # Step 1: Identify Model
        model_type = self.identify_model(question)
        model_name = self.MODEL_DISPLAY_NAMES.get(model_type, "Unknown Inventory Model")

        if model_type == InventoryModelType.UNKNOWN:
            return AgentResponse(
                question=question,
                model_type=model_type,
                model_name=model_name,
                raw_inputs={},
                calculation_result=None,
                verification_passed=False,
                verification_notes=[],
                explanation="Could not unambiguously identify an inventory model. Please specify whether your scenario is an EOQ, Quantity Discount, Safety Stock/ROP, Fixed-Period Review, or Newsvendor problem.",
                error="Unknown inventory model requested.",
            )

        # Step 2: Extract Inputs
        extracted_inputs = self.extract_inputs(question, model_type, custom_inputs)

        # Validate minimum necessary inputs per model
        missing_params = self._validate_inputs(model_type, extracted_inputs)
        if missing_params:
            return AgentResponse(
                question=question,
                model_type=model_type,
                model_name=model_name,
                raw_inputs=extracted_inputs,
                calculation_result=None,
                verification_passed=False,
                verification_notes=[],
                explanation=f"Missing essential parameters for {model_name}: {', '.join(missing_params)}. Please provide these numerical values.",
                error=f"Missing parameters: {', '.join(missing_params)}",
            )

        # Step 3: Select Python Tool
        try:
            tool_fn, tool_path = self.select_tool(model_type)
        except Exception as e:
            return AgentResponse(
                question=question,
                model_type=model_type,
                model_name=model_name,
                raw_inputs=extracted_inputs,
                calculation_result=None,
                verification_passed=False,
                verification_notes=[],
                explanation=f"Tool selection error: {str(e)}",
                error=str(e),
            )

        # Step 4: Calculate (pure Python numerical computation)
        try:
            calc_result = self.calculate(tool_fn, extracted_inputs)
        except Exception as e:
            return AgentResponse(
                question=question,
                model_type=model_type,
                model_name=model_name,
                raw_inputs=extracted_inputs,
                calculation_result=None,
                verification_passed=False,
                verification_notes=[],
                explanation=f"Numerical calculation failed: {str(e)}",
                error=str(e),
            )

        # Step 5: Verify
        passed, notes = self.verify(model_type, calc_result)

        # Step 6: Explain
        explanation = self.explain(model_type, extracted_inputs, calc_result, notes)

        structured = calc_result.to_dict() if hasattr(calc_result, "to_dict") else {}

        return AgentResponse(
            question=question,
            model_type=model_type,
            model_name=model_name,
            raw_inputs=extracted_inputs,
            calculation_result=calc_result,
            verification_passed=passed,
            verification_notes=notes,
            explanation=explanation,
            structured_metrics=structured,
        )

    def _validate_inputs(self, model_type: str, inputs: Dict[str, Any]) -> List[str]:
        """
        Check for mandatory numerical parameters for each model.
        Thoroughly checks all valid mathematical representations before declaring parameters missing.
        """
        missing = []
        if model_type == InventoryModelType.EOQ:
            if inputs.get("demand") is None:
                missing.append("demand (D)")
            if inputs.get("order_cost") is None:
                missing.append("ordering/setup cost (S)")
            if inputs.get("holding_cost") is None and (inputs.get("holding_rate") is None or inputs.get("unit_cost") is None):
                missing.append("holding cost (H) or (holding rate and unit price)")

        elif model_type == InventoryModelType.QUANTITY_DISCOUNT:
            if inputs.get("demand") is None:
                missing.append("demand (D)")
            if inputs.get("order_cost") is None:
                missing.append("ordering/setup cost (S)")
            if inputs.get("holding_rate") is None and inputs.get("fixed_holding_cost") is None:
                missing.append("holding rate (% of price) or fixed holding cost ($/unit/year)")
            if not inputs.get("price_brackets"):
                missing.append("price_brackets schedule (min_qty, max_qty, price)")

        elif model_type == InventoryModelType.SAFETY_STOCK_ROP:
            if inputs.get("daily_demand_mean") is None:
                missing.append("daily_demand_mean (d)")
            if inputs.get("lead_time_mean") is None:
                missing.append("lead_time_mean (L)")
            # If demand or lead time is variable and safety stock is not directly specified, service level or z is required
            if inputs.get("safety_stock_fixed") is None:
                sigma_d = inputs.get("daily_demand_std", 0.0)
                sigma_l = inputs.get("lead_time_std", 0.0)
                if (sigma_d > 0.0 or sigma_l > 0.0) and inputs.get("service_level") is None and inputs.get("z_score") is None:
                    missing.append("service_level (CSL) or z_score")

        elif model_type == InventoryModelType.FIXED_PERIOD_REVIEW:
            if inputs.get("review_period") is None:
                missing.append("review_period (T)")
            if inputs.get("lead_time") is None:
                missing.append("lead_time (L)")
            if inputs.get("daily_demand_mean") is None:
                missing.append("daily_demand_mean (d)")

        elif model_type == InventoryModelType.NEWSVENDOR:
            if inputs.get("demand_mean") is None:
                missing.append("demand_mean (mu)")
            # Can be specified either via (selling_price and purchase_cost) OR via direct (underage_cost_direct and overage_cost_direct)
            has_p_and_c = (inputs.get("selling_price") is not None and inputs.get("purchase_cost") is not None)
            has_cu_and_co = (inputs.get("underage_cost_direct") is not None and inputs.get("overage_cost_direct") is not None)
            if not has_p_and_c and not has_cu_and_co:
                if inputs.get("selling_price") is None:
                    missing.append("selling_price (P) [or underage cost Cu]")
                if inputs.get("purchase_cost") is None:
                    missing.append("purchase_cost (C) [or overage cost Co]")

        return missing
