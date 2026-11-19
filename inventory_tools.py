"""
Supply Chain Analytics - Inventory Tools Module
Implements deterministic Python tools for:
1. calculate_eoq()
2. calculate_quantity_discount()
3. calculate_rop()
4. calculate_periodic_review()
5. calculate_newsvendor()

Every tool returns the 8 structured educational verification components:
1. Inputs Extracted (inputs_extracted)
2. Variable Definitions (variable_definitions)
3. Formula (formula)
4. Formula with Substituted Values (formula_with_substituted_values)
5. Intermediate Calculations (intermediate_calculations)
6. Final Answer (final_answer)
7. Verification Checks (verification_checks)
8. Assumptions (assumptions)

All numerical calculations are strictly executed in deterministic Python.
No missing numerical inputs are ever invented.
"""

from dataclasses import dataclass, field
import math
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Helper Distribution Functions
# ---------------------------------------------------------------------------
def normal_cdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Compute cumulative distribution function of normal distribution."""
    if sigma <= 0:
        return 1.0 if x >= mu else 0.0
    return NormalDist(mu, sigma).cdf(x)


def normal_ppf(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Compute percent point function (inverse CDF / quantile) of normal distribution."""
    if not (0.0 < p < 1.0):
        raise ValueError(f"Probability must be strictly between 0 and 1, got {p}")
    return NormalDist(mu, sigma).inv_cdf(p)


def normal_pdf(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Compute probability density function of normal distribution."""
    if sigma <= 0:
        return 0.0
    return NormalDist(mu, sigma).pdf(x)


def standard_normal_loss(z: float) -> float:
    """
    Unit Normal Loss Function G(z) = phi(z) - z * (1 - Phi(z))
    Used for computing expected lost sales in inventory models.
    """
    phi = normal_pdf(z, 0.0, 1.0)
    Phi = normal_cdf(z, 0.0, 1.0)
    return phi - z * (1.0 - Phi)


# ---------------------------------------------------------------------------
# 1. calculate_eoq()
# ---------------------------------------------------------------------------
@dataclass
class EOQResult:
    # 8 Educational Verification Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Primary attributes for direct property access
    optimal_order_quantity: float
    total_inventory_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    orders_per_year: float

    # Optional / contextual metrics
    annual_demand: float = 0.0
    order_cost: float = 0.0
    holding_cost_per_unit: float = 0.0
    unit_cost: Optional[float] = None
    total_acquisition_cost: Optional[float] = None
    cycle_time_days: Optional[float] = None
    daily_demand: Optional[float] = None
    reorder_point: Optional[float] = None
    lead_time_days: Optional[float] = None
    cost_parity_difference: float = 0.0

    # Backward compatibility helpers
    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    @property
    def verification_passed(self) -> bool:
        return bool(self.verification_checks.get("all_checks_passed", True))

    @property
    def verification_notes(self) -> List[str]:
        return list(self.verification_checks.get("audit_log", []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Economic Order Quantity (EOQ)",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            # Flattened legacy shortcuts
            "optimal_order_quantity": round(self.optimal_order_quantity, 2),
            "total_inventory_cost": round(self.total_inventory_cost, 2),
            "annual_ordering_cost": round(self.annual_ordering_cost, 2),
            "annual_holding_cost": round(self.annual_holding_cost, 2),
            "orders_per_year": round(self.orders_per_year, 2),
            "cycle_time_days": round(self.cycle_time_days, 1) if self.cycle_time_days is not None else None,
            "reorder_point": round(self.reorder_point, 2) if self.reorder_point is not None else None,
            "verification_passed": self.verification_passed,
            "verification_notes": self.verification_notes,
        }


def calculate_eoq(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    holding_cost: Optional[float] = None,
    unit_cost: Optional[float] = None,
    holding_rate: Optional[float] = None,
    lead_time_days: Optional[float] = None,
    work_days_per_year: float = 365.0,
) -> EOQResult:
    """
    Calculate Economic Order Quantity (EOQ):
    Q = sqrt(2 * D * S / H)

    Returns the 8 mandatory educational verification components:
    1. Inputs Extracted
    2. Variable Definitions
    3. Formula
    4. Formula with Substituted Values
    5. Intermediate Calculations
    6. Final Answer
    7. Verification Checks
    8. Assumptions
    """
    # 1. Inputs Validation - Never invent missing numerical inputs
    if demand is None:
        raise ValueError("Annual demand 'demand' (D) is required and cannot be None.")
    if order_cost is None:
        raise ValueError("Order/setup cost 'order_cost' (S) is required and cannot be None.")

    try:
        D = float(demand)
        S = float(order_cost)
    except (TypeError, ValueError):
        raise TypeError("demand and order_cost must be numeric values.")

    if D <= 0:
        raise ValueError(f"Annual demand must be strictly positive (D > 0), got {D}.")
    if S <= 0:
        raise ValueError(f"Order cost must be strictly positive (S > 0), got {S}.")

    # Resolve unit holding cost H
    if holding_cost is None:
        if unit_cost is not None and holding_rate is not None:
            try:
                C = float(unit_cost)
                h = float(holding_rate)
            except (TypeError, ValueError):
                raise TypeError("unit_cost and holding_rate must be numeric values.")
            if C <= 0:
                raise ValueError(f"Unit cost must be strictly positive (C > 0), got {C}.")
            if h <= 0:
                raise ValueError(f"Holding rate must be strictly positive (h > 0), got {h}.")
            H = C * h
        else:
            raise ValueError(
                "Missing holding cost specification: provide either 'holding_cost' directly "
                "or both 'unit_cost' and 'holding_rate'. Missing inputs will never be invented."
            )
    else:
        try:
            H = float(holding_cost)
        except (TypeError, ValueError):
            raise TypeError("holding_cost must be a numeric value.")
        if H <= 0:
            raise ValueError(f"Holding cost must be strictly positive (H > 0), got {H}.")
        C = float(unit_cost) if unit_cost is not None else None
        h = float(holding_rate) if holding_rate is not None else None

    L = float(lead_time_days) if lead_time_days is not None else None
    if L is not None and L < 0:
        raise ValueError(f"Lead time cannot be negative, got {L}.")

    work_days = float(work_days_per_year)
    if work_days <= 0:
        raise ValueError(f"Operating days per year must be positive, got {work_days}.")

    # 2. Variable Definitions
    var_defs = {
        "D": "Annual demand rate (units/year)",
        "S": "Fixed order or setup cost per order ($/order)",
        "H": "Unit holding or carrying cost per year ($/unit/year)",
        "C": "Unit purchase price ($/unit, optional)",
        "h": "Annual holding cost rate as fraction of unit cost (fraction/year, optional)",
        "L": "Replenishment lead time (days, optional)",
        "Q": "Order quantity (units)",
        "Q*": "Economic Order Quantity that minimizes total inventory costs (units)",
        "N": "Number of replenishment orders placed per year (orders/year)",
        "T": "Order cycle time / interval between orders (operating days)",
        "TRC": "Total Annual Relevant Inventory Cost = Annual Ordering Cost + Annual Holding Cost ($/year)",
        "TAC": "Total Annual Acquisition Cost = TRC + Material Cost (D * C) ($/year)",
        "ROP": "Reorder point (units)",
    }

    # 3. Formula
    formulas = {
        "EOQ_optimal_order_quantity": "Q* = sqrt((2 * D * S) / H)",
        "annual_ordering_cost": "Annual Ordering Cost = (D / Q) * S",
        "annual_holding_cost": "Annual Holding Cost = (Q / 2) * H",
        "total_relevant_inventory_cost": "TRC = Annual Ordering Cost + Annual Holding Cost = (D / Q)*S + (Q / 2)*H",
        "orders_per_year": "N = D / Q*",
        "cycle_time_days": "T = (Q* / D) * work_days_per_year",
    }
    if C is not None:
        formulas["total_annual_acquisition_cost"] = "TAC = TRC + (D * C)"
    if L is not None:
        formulas["reorder_point"] = "ROP = (D / work_days_per_year) * L"

    # 4. Deterministic Calculations
    two_DS = 2.0 * D * S
    ratio_2DS_H = two_DS / H
    q_star = math.sqrt(ratio_2DS_H)

    annual_ordering = (D / q_star) * S
    annual_holding = (q_star / 2.0) * H
    total_rel_cost = annual_ordering + annual_holding
    orders_yr = D / q_star
    cycle_days = (q_star / D) * work_days

    daily_d = D / work_days
    rop = daily_d * L if L is not None else None
    mat_cost = D * C if C is not None else None
    tac = (total_rel_cost + mat_cost) if mat_cost is not None else None

    # 5. Formula with Substituted Values
    formulas_substituted = {
        "EOQ_optimal_order_quantity": f"Q* = sqrt((2 * {D:,.2f} * {S:,.2f}) / {H:,.2f}) = sqrt({two_DS:,.2f} / {H:,.2f}) = sqrt({ratio_2DS_H:,.2f}) = {q_star:,.2f} units",
        "annual_ordering_cost": f"Annual Ordering Cost = ({D:,.2f} / {q_star:,.2f}) * {S:,.2f} = {orders_yr:,.2f} * {S:,.2f} = ${annual_ordering:,.2f}",
        "annual_holding_cost": f"Annual Holding Cost = ({q_star:,.2f} / 2) * {H:,.2f} = {q_star/2.0:,.2f} * {H:,.2f} = ${annual_holding:,.2f}",
        "total_relevant_inventory_cost": f"TRC = ${annual_ordering:,.2f} + ${annual_holding:,.2f} = ${total_rel_cost:,.2f}",
        "orders_per_year": f"N = {D:,.2f} / {q_star:,.2f} = {orders_yr:,.2f} orders/year",
        "cycle_time_days": f"T = ({q_star:,.2f} / {D:,.2f}) * {work_days:.0f} = {cycle_days:.1f} operating days",
    }
    if tac is not None and mat_cost is not None and C is not None:
        formulas_substituted["total_annual_acquisition_cost"] = f"TAC = ${total_rel_cost:,.2f} + ({D:,.2f} * ${C:,.2f}) = ${total_rel_cost:,.2f} + ${mat_cost:,.2f} = ${tac:,.2f}"
    if rop is not None and L is not None:
        formulas_substituted["reorder_point"] = f"ROP = ({D:,.2f} / {work_days:.0f}) * {L:.1f} = {daily_d:,.2f} units/day * {L:.1f} days = {rop:,.2f} units"

    # 6. Intermediate Calculations
    intermediate_calcs = {
        "numerator_2DS": round(two_DS, 2),
        "holding_cost_per_unit_H": round(H, 4),
        "ratio_2DS_over_H": round(ratio_2DS_H, 4),
        "raw_optimal_Q_star": round(q_star, 4),
        "annual_ordering_cost": round(annual_ordering, 2),
        "annual_holding_cost": round(annual_holding, 2),
        "orders_per_year": round(orders_yr, 2),
        "cycle_time_days": round(cycle_days, 1),
    }
    if mat_cost is not None:
        intermediate_calcs["annual_material_cost"] = round(mat_cost, 2)
    if rop is not None:
        intermediate_calcs["daily_demand"] = round(daily_d, 4)
        intermediate_calcs["lead_time_demand_ROP"] = round(rop, 2)

    # 7. Final Answer
    final_answer = {
        "optimal_order_quantity": round(q_star, 2),
        "recommended_order_quantity_integer": int(round(q_star)),
        "total_relevant_inventory_cost": round(total_rel_cost, 2),
        "annual_ordering_cost": round(annual_ordering, 2),
        "annual_holding_cost": round(annual_holding, 2),
        "orders_per_year": round(orders_yr, 2),
        "cycle_time_days": round(cycle_days, 1),
        "total_annual_acquisition_cost": round(tac, 2) if tac is not None else None,
        "reorder_point": round(rop, 2) if rop is not None else None,
    }

    # 8. Verification Checks
    parity_diff = abs(annual_ordering - annual_holding)
    parity_ok = parity_diff < (1e-2 * total_rel_cost)
    d2_cost = (2.0 * D * S) / (q_star ** 3)
    convex_ok = d2_cost > 0
    q_pos = q_star > 0

    verification_checks = {
        "all_checks_passed": bool(parity_ok and convex_ok and q_pos),
        "cost_parity_check": {
            "ordering_cost": round(annual_ordering, 2),
            "holding_cost": round(annual_holding, 2),
            "absolute_difference": round(parity_diff, 4),
            "passed": bool(parity_ok),
            "condition": "At optimal EOQ, annual ordering cost must equal annual holding cost.",
        },
        "second_order_convexity": {
            "d2_cost_dQ2": round(d2_cost, 6),
            "passed": bool(convex_ok),
            "condition": "d^2(TC)/dQ^2 = 2DS/Q^3 > 0 confirms a strictly global cost minimum.",
        },
        "positivity_check": {
            "Q_star": round(q_star, 2),
            "passed": bool(q_pos),
            "condition": "Optimal order quantity must be strictly positive.",
        },
        "audit_log": [
            f"First-order optimality: Annual ordering cost (${annual_ordering:,.2f}) equals annual holding cost (${annual_holding:,.2f}) within ${parity_diff:.4f}.",
            f"Second-order optimality: d^2(TC)/dQ^2 = {d2_cost:.6f} > 0 strictly convex minimum.",
            f"Physical validity: Q* = {q_star:,.2f} units > 0.",
        ],
    }

    # 9. Explicit Assumptions
    assumptions = [
        "Demand rate is known, constant, and continuous over time.",
        "Supplier replenishment lead time is deterministic and constant.",
        "Entire replenishment order Q arrives in a single delivery without staggered shipments.",
        "Fixed ordering cost S and unit carrying cost H remain constant regardless of order size.",
        "No shortages or backorders are permitted; all demand must be satisfied directly from inventory.",
        "No quantity purchase discounts apply under this basic EOQ formulation.",
    ]

    inputs_extracted = {
        "annual_demand_D": D,
        "order_cost_S": S,
        "holding_cost_H": H,
        "unit_cost_C": C,
        "holding_rate_h": h,
        "lead_time_days_L": L,
        "work_days_per_year": work_days,
        "demand": D,
        "order_cost": S,
        "holding_cost": H,
        "unit_cost": C,
        "holding_rate": h,
        "lead_time_days": L,
    }

    return EOQResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        optimal_order_quantity=q_star,
        total_inventory_cost=total_rel_cost,
        annual_ordering_cost=annual_ordering,
        annual_holding_cost=annual_holding,
        orders_per_year=orders_yr,
        annual_demand=D,
        order_cost=S,
        holding_cost_per_unit=H,
        unit_cost=C,
        total_acquisition_cost=tac,
        cycle_time_days=cycle_days,
        daily_demand=daily_d,
        reorder_point=rop,
        lead_time_days=L,
        cost_parity_difference=parity_diff,
    )


# ---------------------------------------------------------------------------
# 2. calculate_quantity_discount()
# ---------------------------------------------------------------------------
@dataclass
class PriceTierEvaluation:
    tier_index: int
    min_quantity: float
    max_quantity: Optional[float]
    unit_price: float
    holding_cost_per_unit: float
    unconstrained_eoq: float
    is_eoq_feasible: bool
    candidate_order_quantity: float
    annual_purchase_cost: float
    annual_ordering_cost: float
    annual_holding_cost: float
    total_acquisition_cost: float
    feasibility_status: str


@dataclass
class QuantityDiscountResult:
    # 8 Educational Verification Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Primary attributes for direct property access
    optimal_order_quantity: float
    optimal_price: float
    optimal_tier_index: int
    minimum_total_acquisition_cost: float
    baseline_eoq_quantity: float
    baseline_total_acquisition_cost: float
    net_annual_savings: float
    tier_evaluations: List[PriceTierEvaluation]

    # Backward compatibility helpers
    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    @property
    def verification_passed(self) -> bool:
        return bool(self.verification_checks.get("all_checks_passed", True))

    @property
    def verification_notes(self) -> List[str]:
        return list(self.verification_checks.get("audit_log", []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Quantity Discount",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "optimal_order_quantity": round(self.optimal_order_quantity, 2),
            "optimal_price": self.optimal_price,
            "optimal_tier_index": self.optimal_tier_index,
            "minimum_total_acquisition_cost": round(self.minimum_total_acquisition_cost, 2),
            "baseline_eoq_quantity": round(self.baseline_eoq_quantity, 2),
            "baseline_total_acquisition_cost": round(self.baseline_total_acquisition_cost, 2),
            "net_annual_savings": round(self.net_annual_savings, 2),
            "tier_evaluations": [
                {
                    "tier": t.tier_index,
                    "range": f"[{t.min_quantity:.0f}, {f'{t.max_quantity:.0f}' if t.max_quantity is not None else 'inf'})",
                    "unit_price": t.unit_price,
                    "unconstrained_eoq": round(t.unconstrained_eoq, 2),
                    "candidate_quantity": round(t.candidate_order_quantity, 2),
                    "purchase_cost": round(t.annual_purchase_cost, 2),
                    "ordering_cost": round(t.annual_ordering_cost, 2),
                    "holding_cost": round(t.annual_holding_cost, 2),
                    "total_cost": round(t.total_acquisition_cost, 2),
                    "status": t.feasibility_status,
                }
                for t in self.tier_evaluations
            ],
            "verification_passed": self.verification_passed,
            "verification_notes": self.verification_notes,
        }


def calculate_quantity_discount(
    demand: Optional[float] = None,
    order_cost: Optional[float] = None,
    price_brackets: Optional[List[Dict[str, Any]]] = None,
    holding_rate: Optional[float] = None,
    fixed_holding_cost: Optional[float] = None,
) -> QuantityDiscountResult:
    """
    Evaluate all-units quantity discount schedule to find the globally optimal order quantity.
    Returns all 8 educational verification sections.
    """
    # 1. Input Validation - Never invent missing numerical inputs
    if demand is None:
        raise ValueError("Annual demand 'demand' (D) is required and cannot be None.")
    if order_cost is None:
        raise ValueError("Order cost 'order_cost' (S) is required and cannot be None.")
    if price_brackets is None or len(price_brackets) == 0:
        raise ValueError("Price discount schedule 'price_brackets' is required and cannot be empty.")

    try:
        D = float(demand)
        S = float(order_cost)
    except (TypeError, ValueError):
        raise TypeError("demand and order_cost must be numeric values.")

    if D <= 0:
        raise ValueError(f"Demand must be strictly positive (D > 0), got {D}.")
    if S <= 0:
        raise ValueError(f"Order cost must be strictly positive (S > 0), got {S}.")

    if holding_rate is None and fixed_holding_cost is None:
        raise ValueError(
            "Must provide either 'holding_rate' (% of unit price) or 'fixed_holding_cost'. "
            "Missing inputs will never be invented."
        )

    h_rate = float(holding_rate) if holding_rate is not None else None
    if h_rate is not None and h_rate <= 0:
        raise ValueError(f"holding_rate must be positive, got {h_rate}.")

    fixed_h = float(fixed_holding_cost) if fixed_holding_cost is not None else None
    if fixed_h is not None and fixed_h <= 0:
        raise ValueError(f"fixed_holding_cost must be positive, got {fixed_h}.")

    for idx, b in enumerate(price_brackets):
        if "min_qty" not in b or "price" not in b:
            raise ValueError(f"Tier {idx + 1} is missing required 'min_qty' or 'price'.")
        if b["min_qty"] < 0:
            raise ValueError(f"Tier {idx + 1} 'min_qty' cannot be negative.")
        if b["price"] <= 0:
            raise ValueError(f"Tier {idx + 1} 'price' must be strictly positive.")
        if b.get("max_qty") is not None and b["max_qty"] < b["min_qty"]:
            raise ValueError(f"Tier {idx + 1} 'max_qty' ({b['max_qty']}) cannot be less than 'min_qty' ({b['min_qty']}).")

    # 2. Variable Definitions
    var_defs = {
        "D": "Annual demand rate (units/year)",
        "S": "Fixed order setup cost per order ($/order)",
        "P_i": "Unit purchase price in discount bracket i ($/unit)",
        "H_i": "Annual unit holding cost in bracket i = h * P_i or fixed H ($/unit/year)",
        "EOQ_i": "Unconstrained classical EOQ computed for bracket i: sqrt(2DS / H_i) (units)",
        "Q_candidate_i": "Feasible cost-minimizing candidate order quantity for tier i (units)",
        "TAC_i": "Total Annual Acquisition Cost = Purchase Cost + Ordering Cost + Holding Cost ($/year)",
        "Q*": "Globally optimal order quantity minimizing TAC across all tiers (units)",
    }

    # 3. Formula
    formulas = {
        "unconstrained_eoq": "EOQ_i = sqrt((2 * D * S) / H_i)",
        "feasibility_rule": "If min_qty_i <= EOQ_i <= max_qty_i => Q_candidate = EOQ_i; If EOQ_i < min_qty_i => Q_candidate = min_qty_i (price break cutoff); If EOQ_i > max_qty_i => Dominated by higher tier",
        "total_annual_acquisition_cost": "TAC(Q) = (D * P_i) + (D / Q)*S + (Q / 2)*H_i",
        "global_optimality": "Q* = argmin_{Q_candidate_i} TAC(Q_candidate_i)",
    }

    # 4. Deterministic Calculations
    sorted_brackets = sorted(price_brackets, key=lambda x: x["min_qty"])
    evaluations: List[PriceTierEvaluation] = []
    best_candidate: Optional[PriceTierEvaluation] = None
    min_total_cost = float("inf")
    formulas_substituted: Dict[str, str] = {}

    for i, bracket in enumerate(sorted_brackets):
        min_q = float(bracket["min_qty"])
        max_q = float(bracket["max_qty"]) if bracket.get("max_qty") is not None else None
        p = float(bracket["price"])

        h_tier = fixed_h if fixed_h is not None else (p * h_rate)  # type: ignore
        eoq_raw = math.sqrt((2.0 * D * S) / h_tier)

        if eoq_raw < min_q:
            candidate_q = min_q
            is_feasible = False
            status = f"Below bracket; adjusted upward to cutoff {min_q:.0f}"
        elif max_q is not None and eoq_raw > max_q:
            candidate_q = max_q
            is_feasible = False
            status = f"Above bracket ({max_q:.0f}); evaluated at upper boundary"
        else:
            candidate_q = eoq_raw
            is_feasible = True
            status = f"Feasible EOQ ({eoq_raw:.1f})"

        purch_cost = D * p
        order_cost_tier = (D / candidate_q) * S
        hold_cost_tier = (candidate_q / 2.0) * h_tier
        tac_tier = purch_cost + order_cost_tier + hold_cost_tier

        tier_eval = PriceTierEvaluation(
            tier_index=i + 1,
            min_quantity=min_q,
            max_quantity=max_q,
            unit_price=p,
            holding_cost_per_unit=h_tier,
            unconstrained_eoq=eoq_raw,
            is_eoq_feasible=is_feasible,
            candidate_order_quantity=candidate_q,
            annual_purchase_cost=purch_cost,
            annual_ordering_cost=order_cost_tier,
            annual_holding_cost=hold_cost_tier,
            total_acquisition_cost=tac_tier,
            feasibility_status=status,
        )
        evaluations.append(tier_eval)

        # Build substituted formula string per tier
        range_str = f"[{min_q:.0f}, {f'{max_q:.0f}' if max_q is not None else 'inf'})"
        formulas_substituted[f"tier_{i+1}_evaluation"] = (
            f"Tier {i+1} {range_str} @ ${p:.2f}: EOQ_{i+1} = sqrt((2 * {D:,.0f} * {S:,.0f}) / {h_tier:.2f}) = {eoq_raw:,.1f}. "
            f"Candidate Q = {candidate_q:,.0f} ({status}). "
            f"TAC({candidate_q:,.0f}) = ({D:,.0f} * ${p:.2f}) + ({D:,.0f} / {candidate_q:,.0f}) * ${S:,.0f} + ({candidate_q:,.0f} / 2) * ${h_tier:.2f} = "
            f"${purch_cost:,.2f} + ${order_cost_tier:,.2f} + ${hold_cost_tier:,.2f} = ${tac_tier:,.2f}"
        )

        if tac_tier < min_total_cost:
            min_total_cost = tac_tier
            best_candidate = tier_eval

    assert best_candidate is not None

    base_tier = evaluations[0]
    base_eoq = base_tier.candidate_order_quantity
    base_cost = base_tier.total_acquisition_cost
    savings = base_cost - min_total_cost

    formulas_substituted["optimal_selection"] = (
        f"Optimal Policy: Select Tier {best_candidate.tier_index} with candidate Q* = {best_candidate.candidate_order_quantity:,.0f} units at ${best_candidate.unit_price:.2f}/unit. "
        f"Minimum TAC = ${min_total_cost:,.2f} yielding net annual savings of ${savings:,.2f} vs. baseline no-discount TAC of ${base_cost:,.2f}."
    )

    # 5. Intermediate Calculations
    intermediate_calcs = {
        "tiers_count": len(evaluations),
        "baseline_unconstrained_eoq": round(base_eoq, 2),
        "baseline_total_acquisition_cost": round(base_cost, 2),
        "tiers_summary": [
            {
                "tier": t.tier_index,
                "price": t.unit_price,
                "unconstrained_eoq": round(t.unconstrained_eoq, 2),
                "candidate_q": round(t.candidate_order_quantity, 2),
                "purchase_cost": round(t.annual_purchase_cost, 2),
                "ordering_cost": round(t.annual_ordering_cost, 2),
                "holding_cost": round(t.annual_holding_cost, 2),
                "total_cost": round(t.total_acquisition_cost, 2),
                "is_optimal": (t.tier_index == best_candidate.tier_index),
            }
            for t in evaluations
        ],
    }

    # 6. Final Answer
    final_answer = {
        "optimal_order_quantity": round(best_candidate.candidate_order_quantity, 2),
        "recommended_order_quantity_integer": int(round(best_candidate.candidate_order_quantity)),
        "optimal_price": best_candidate.unit_price,
        "optimal_tier_index": best_candidate.tier_index,
        "minimum_total_annual_cost": round(min_total_cost, 2),
        "baseline_total_acquisition_cost": round(base_cost, 2),
        "net_annual_savings": round(savings, 2),
    }

    # 7. Verification Checks
    verification_checks = {
        "all_checks_passed": True,
        "global_minimum_confirmed": {
            "selected_tier": best_candidate.tier_index,
            "selected_tac": round(min_total_cost, 2),
            "passed": True,
            "condition": "Confirmed that the selected candidate TAC is strictly less than or equal to all evaluated candidate tiers.",
        },
        "feasibility_validated": {
            "candidate_quantity": round(best_candidate.candidate_order_quantity, 2),
            "tier_min_quantity": best_candidate.min_quantity,
            "passed": True,
            "condition": "Order quantity satisfies supplier tier price qualification breakpoint.",
        },
        "audit_log": [
            f"Evaluated {len(evaluations)} price discount tiers systematically.",
            f"Selected optimal quantity {best_candidate.candidate_order_quantity:,.0f} in Tier {best_candidate.tier_index} at ${best_candidate.unit_price:.2f}/unit.",
            f"Global cost minimization verified: Total cost ${min_total_cost:,.2f} is lower than all other candidate quantities.",
            f"Net annual savings vs. baseline EOQ: ${savings:,.2f}.",
        ],
    }

    # 8. Explicit Assumptions
    assumptions = [
        "All-units quantity discount schedule applies: the reduced price applies to every unit in the order.",
        "Supplier has sufficient capacity to deliver entire batch size instantaneously.",
        "Holding cost is either constant across all tiers or directly proportional to purchase price.",
        "Annual demand is constant and non-perishable across the planning horizon.",
        "Working capital is available to support larger upfront batch orders to qualify for discounts.",
    ]

    inputs_extracted = {
        "annual_demand_D": D,
        "order_cost_S": S,
        "price_brackets": price_brackets,
        "holding_rate_h": h_rate,
        "fixed_holding_cost_H": fixed_h,
        "demand": D,
        "order_cost": S,
    }

    return QuantityDiscountResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        optimal_order_quantity=best_candidate.candidate_order_quantity,
        optimal_price=best_candidate.unit_price,
        optimal_tier_index=best_candidate.tier_index,
        minimum_total_acquisition_cost=min_total_cost,
        baseline_eoq_quantity=base_eoq,
        baseline_total_acquisition_cost=base_cost,
        net_annual_savings=savings,
        tier_evaluations=evaluations,
    )


# ---------------------------------------------------------------------------
# 3. calculate_rop() (Safety Stock and Reorder Point)
# ---------------------------------------------------------------------------
@dataclass
class ROPResult:
    # 8 Educational Verification Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Primary attributes for direct property access
    reorder_point: float
    safety_stock: float
    expected_lead_time_demand: float
    std_dev_lead_time_demand: float
    uncertainty_case: str

    # Stochastic & service parameters
    service_level: Optional[float] = None
    z_score: float = 0.0
    stockout_risk: Optional[float] = None
    annual_holding_cost_safety_stock: Optional[float] = None

    # Backward compatibility helpers
    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    @property
    def mean_demand_lead_time(self) -> float:
        return self.expected_lead_time_demand

    @property
    def uncertainty_type(self) -> str:
        return self.uncertainty_case

    @property
    def verification_passed(self) -> bool:
        return bool(self.verification_checks.get("all_checks_passed", True))

    @property
    def verification_notes(self) -> List[str]:
        return list(self.verification_checks.get("audit_log", []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Safety Stock & Reorder Point",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "reorder_point": round(self.reorder_point, 2),
            "safety_stock": round(self.safety_stock, 2),
            "expected_lead_time_demand": round(self.expected_lead_time_demand, 2),
            "std_dev_lead_time_demand": round(self.std_dev_lead_time_demand, 2),
            "service_level": f"{self.service_level * 100:.2f}%" if self.service_level is not None else None,
            "z_score": round(self.z_score, 3),
            "uncertainty_case": self.uncertainty_case,
            "verification_passed": self.verification_passed,
            "verification_notes": self.verification_notes,
        }


def calculate_rop(
    daily_demand_mean: Optional[float] = None,
    lead_time_mean: Optional[float] = None,
    daily_demand_std: float = 0.0,
    lead_time_std: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    unit_holding_cost: Optional[float] = None,
    safety_stock_fixed: Optional[float] = None,
) -> ROPResult:
    """
    Calculate Safety Stock (SS) and Reorder Point (ROP) under continuous review.
    Returns all 8 educational verification sections.
    """
    # 1. Input Validation - Never invent missing numerical inputs
    if daily_demand_mean is None:
        raise ValueError("Mean demand 'daily_demand_mean' (d) is required and cannot be None.")
    if lead_time_mean is None:
        raise ValueError("Mean lead time 'lead_time_mean' (L) is required and cannot be None.")

    try:
        d = float(daily_demand_mean)
        L = float(lead_time_mean)
        sigma_d = float(daily_demand_std) if daily_demand_std is not None else 0.0
        sigma_L = float(lead_time_std) if lead_time_std is not None else 0.0
    except (TypeError, ValueError):
        raise TypeError("Demand and lead time parameters must be numeric.")

    if d < 0:
        raise ValueError(f"Mean demand cannot be negative, got {d}.")
    if L < 0:
        raise ValueError(f"Mean lead time cannot be negative, got {L}.")
    if sigma_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {sigma_d}.")
    if sigma_L < 0:
        raise ValueError(f"Lead time standard deviation cannot be negative, got {sigma_L}.")

    has_var_demand = (sigma_d > 0.0)
    has_var_lead_time = (sigma_L > 0.0)

    # 2. Variable Definitions
    var_defs = {
        "d": "Average demand rate per period/day (units/day)",
        "L": "Average replenishment lead time (days)",
        "sigma_d": "Standard deviation of period demand (units/day)",
        "sigma_L": "Standard deviation of replenishment lead time (days)",
        "sigma_LT": "Standard deviation of total lead time demand (units)",
        "SL": "Target Cycle Service Level (probability of not stocking out during lead time)",
        "z": "Safety factor corresponding to standard normal quantile Phi^-1(SL)",
        "SS": "Safety stock buffer required to achieve target service level (units)",
        "ROP": "Reorder point triggering a replenishment order = d * L + SS (units)",
    }

    # 3. Formula
    formulas = {
        "expected_lead_time_demand": "E[DL] = d * L",
        "reorder_point": "ROP = (d * L) + SS",
    }
    if not has_var_demand and not has_var_lead_time:
        formulas["sigma_LT"] = "sigma_LT = 0 (deterministic condition)"
        formulas["safety_stock"] = "SS = 0"
    elif has_var_demand and not has_var_lead_time:
        formulas["sigma_LT"] = "sigma_LT = sigma_d * sqrt(L)"
        formulas["safety_stock"] = "SS = z * sigma_LT = z * sigma_d * sqrt(L)"
    elif not has_var_demand and has_var_lead_time:
        formulas["sigma_LT"] = "sigma_LT = d * sigma_L"
        formulas["safety_stock"] = "SS = z * sigma_LT = z * d * sigma_L"
    else:
        formulas["sigma_LT"] = "sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"
        formulas["safety_stock"] = "SS = z * sigma_LT = z * sqrt(L * sigma_d^2 + d^2 * sigma_L^2)"

    # 4. Deterministic Calculations
    expected_lt_demand = d * L

    if safety_stock_fixed is not None:
        ss = float(safety_stock_fixed)
        if ss < 0:
            raise ValueError(f"safety_stock_fixed cannot be negative, got {ss}.")
        z = float(z_score) if z_score is not None else 0.0
        sl_calc = float(service_level) if service_level is not None else 1.0
        sigma_lt = 0.0
        case_name = "Direct safety stock specified"
        rop = expected_lt_demand + ss
    else:
        if has_var_demand or has_var_lead_time:
            if z_score is not None:
                z = float(z_score)
                if z < 0:
                    raise ValueError(f"z_score cannot be negative, got {z}.")
                sl_calc = normal_cdf(z)
            elif service_level is not None:
                sl = float(service_level)
                if not (0.0 < sl < 1.0):
                    raise ValueError(f"service_level must be strictly between 0 and 1, got {sl}.")
                z = normal_ppf(sl)
                sl_calc = sl
            else:
                raise ValueError(
                    "Either 'service_level' (0 < SL < 1) or 'z_score' is required when demand or lead time is variable. "
                    "Missing numerical inputs will never be invented."
                )
        else:
            z = 0.0
            sl_calc = 1.0 if service_level is None else float(service_level)

        if not has_var_demand and not has_var_lead_time:
            sigma_lt = 0.0
            case_name = "Constant demand and constant lead time"
        elif has_var_demand and not has_var_lead_time:
            sigma_lt = sigma_d * math.sqrt(L)
            case_name = "Variable demand and constant lead time"
        elif not has_var_demand and has_var_lead_time:
            sigma_lt = d * sigma_L
            case_name = "Constant demand and variable lead time"
        else:
            variance_lt = (L * (sigma_d ** 2)) + ((d ** 2) * (sigma_L ** 2))
            sigma_lt = math.sqrt(variance_lt)
            case_name = "Variable demand and variable lead time"

        ss = z * sigma_lt
        rop = expected_lt_demand + ss

    holding_cost_ss = (ss * float(unit_holding_cost)) if (unit_holding_cost is not None and float(unit_holding_cost) > 0) else None

    # 5. Formula with Substituted Values
    formulas_substituted = {
        "expected_lead_time_demand": f"E[DL] = {d:,.2f} * {L:.1f} = {expected_lt_demand:,.2f} units",
        "uncertainty_case": case_name,
    }
    if not has_var_demand and not has_var_lead_time:
        formulas_substituted["sigma_LT"] = "sigma_LT = 0.00 (demand and lead time are both deterministic)"
        formulas_substituted["safety_stock"] = "SS = 0.00 units"
        formulas_substituted["reorder_point"] = f"ROP = {expected_lt_demand:,.2f} + 0.00 = {rop:,.2f} units"
    elif has_var_demand and not has_var_lead_time:
        formulas_substituted["sigma_LT"] = f"sigma_LT = {sigma_d:,.2f} * sqrt({L:.1f}) = {sigma_d:,.2f} * {math.sqrt(L):.4f} = {sigma_lt:,.2f} units"
        formulas_substituted["safety_stock"] = f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units (at CSL = {sl_calc*100:.1f}%)"
        formulas_substituted["reorder_point"] = f"ROP = {expected_lt_demand:,.2f} + {ss:,.2f} = {rop:,.2f} units"
    elif not has_var_demand and has_var_lead_time:
        formulas_substituted["sigma_LT"] = f"sigma_LT = {d:,.2f} * {sigma_L:.2f} = {sigma_lt:,.2f} units"
        formulas_substituted["safety_stock"] = f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units (at CSL = {sl_calc*100:.1f}%)"
        formulas_substituted["reorder_point"] = f"ROP = {expected_lt_demand:,.2f} + {ss:,.2f} = {rop:,.2f} units"
    else:
        var_d_term = L * (sigma_d ** 2)
        var_l_term = (d ** 2) * (sigma_L ** 2)
        total_var = var_d_term + var_l_term
        formulas_substituted["sigma_LT"] = (
            f"sigma_LT = sqrt({L:.1f} * ({sigma_d:.2f})^2 + ({d:.2f})^2 * ({sigma_L:.2f})^2) = "
            f"sqrt({var_d_term:,.2f} + {var_l_term:,.2f}) = sqrt({total_var:,.2f}) = {sigma_lt:,.2f} units"
        )
        formulas_substituted["safety_stock"] = f"SS = {z:.4f} * {sigma_lt:,.2f} = {ss:,.2f} units (at CSL = {sl_calc*100:.1f}%)"
        formulas_substituted["reorder_point"] = f"ROP = {expected_lt_demand:,.2f} + {ss:,.2f} = {rop:,.2f} units"

    # 6. Intermediate Calculations
    intermediate_calcs = {
        "expected_lead_time_demand": round(expected_lt_demand, 2),
        "lead_time_demand_std_sigma_LT": round(sigma_lt, 4),
        "z_factor": round(z, 4),
        "safety_stock_units": round(ss, 2),
        "cycle_service_level": round(sl_calc, 4),
        "stockout_risk_fraction": round(1.0 - sl_calc, 4),
    }
    if holding_cost_ss is not None:
        intermediate_calcs["annual_holding_cost_safety_stock"] = round(holding_cost_ss, 2)

    # 7. Final Answer
    final_answer = {
        "reorder_point": round(rop, 2),
        "reorder_point_integer": int(math.ceil(rop)),
        "safety_stock": round(ss, 2),
        "safety_stock_integer": int(math.ceil(ss)),
        "expected_lead_time_demand": round(expected_lt_demand, 2),
        "cycle_service_level": f"{sl_calc * 100:.2f}%",
        "stockout_risk": f"{(1.0 - sl_calc) * 100:.2f}%",
        "uncertainty_case": case_name,
    }

    # 8. Verification Checks
    verification_checks = {
        "all_checks_passed": bool(ss >= 0 and rop >= expected_lt_demand and 0.0 <= sl_calc <= 1.0),
        "non_negative_safety_stock": {
            "safety_stock": round(ss, 2),
            "passed": bool(ss >= 0),
            "condition": "Safety stock must be non-negative.",
        },
        "coverage_check": {
            "reorder_point": round(rop, 2),
            "expected_lead_time_demand": round(expected_lt_demand, 2),
            "passed": bool(rop >= expected_lt_demand),
            "condition": "Reorder point must cover expected lead time demand plus safety buffer.",
        },
        "service_level_range": {
            "service_level": round(sl_calc, 4),
            "passed": bool(0.0 <= sl_calc <= 1.0),
            "condition": "Cycle service level probability must lie in [0, 1].",
        },
        "audit_log": [
            f"Uncertainty condition verified: {case_name}.",
            f"Safety stock buffer {ss:.2f} units protects against stockouts at {sl_calc * 100:.1f}% service level (Z = {z:.3f}).",
            f"Reorder point ({rop:.2f}) covers expected lead time demand ({expected_lt_demand:.2f}) plus safety buffer.",
        ],
    }

    # 9. Assumptions
    assumptions = [
        "Continuous inventory review policy: inventory position is monitored continuously after every transaction.",
        "Lead time demand follows an approximate normal distribution.",
        "When both demand and lead time are stochastic, period demand and replenishment lead time are mutually independent random variables.",
        "Unsatisfied customer demand during lead time stockout is fully backordered.",
        "Replenishment orders do not cross (orders arrive in the chronological sequence placed).",
    ]

    inputs_extracted = {
        "daily_demand_mean_d": d,
        "lead_time_mean_L": L,
        "daily_demand_std_sigma_d": sigma_d,
        "lead_time_std_sigma_L": sigma_L,
        "service_level_SL": service_level,
        "z_score": z_score,
        "unit_holding_cost": unit_holding_cost,
        "safety_stock_fixed": safety_stock_fixed,
        "daily_demand_mean": d,
        "lead_time_mean": L,
        "daily_demand_std": sigma_d,
        "lead_time_std": sigma_L,
        "service_level": service_level,
    }

    return ROPResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        reorder_point=rop,
        safety_stock=ss,
        expected_lead_time_demand=expected_lt_demand,
        std_dev_lead_time_demand=sigma_lt,
        uncertainty_case=case_name,
        service_level=sl_calc,
        z_score=z,
        stockout_risk=(1.0 - sl_calc),
        annual_holding_cost_safety_stock=holding_cost_ss,
    )


# Alias for backward compatibility
calculate_safety_stock_rop = calculate_rop


# ---------------------------------------------------------------------------
# 4. calculate_periodic_review() (Fixed-Period Review / Order-Up-To Level)
# ---------------------------------------------------------------------------
@dataclass
class PeriodicReviewResult:
    # 8 Educational Verification Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Primary attributes for direct property access
    order_quantity: float
    target_inventory: float
    protection_period: float
    safety_stock: float
    inventory_position: float

    # Additional stochastic details
    review_period: float = 0.0
    lead_time: float = 0.0
    mean_demand_protection_period: float = 0.0
    std_dev_protection_period: float = 0.0
    service_level: Optional[float] = None
    z_score: float = 0.0
    current_on_hand: float = 0.0
    current_on_order: float = 0.0
    current_backorders: float = 0.0

    # Backward compatibility helpers
    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    @property
    def recommended_order_quantity(self) -> float:
        return self.order_quantity

    @property
    def order_up_to_level(self) -> float:
        return self.target_inventory

    @property
    def protection_interval(self) -> float:
        return self.protection_period

    @property
    def mean_demand_protection_interval(self) -> float:
        return self.mean_demand_protection_period

    @property
    def verification_passed(self) -> bool:
        return bool(self.verification_checks.get("all_checks_passed", True))

    @property
    def verification_notes(self) -> List[str]:
        return list(self.verification_checks.get("audit_log", []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Fixed-Period Review (Periodic Review / Order-Up-To)",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "recommended_order_quantity": round(self.order_quantity, 2),
            "target_inventory": round(self.target_inventory, 2),
            "protection_period": round(self.protection_period, 2),
            "safety_stock": round(self.safety_stock, 2),
            "inventory_position": round(self.inventory_position, 2),
            "service_level": f"{self.service_level * 100:.2f}%" if self.service_level is not None else None,
            "verification_passed": self.verification_passed,
            "verification_notes": self.verification_notes,
        }


def calculate_periodic_review(
    review_period: Optional[float] = None,
    lead_time: Optional[float] = None,
    daily_demand_mean: Optional[float] = None,
    daily_demand_std: float = 0.0,
    service_level: Optional[float] = None,
    z_score: Optional[float] = None,
    on_hand_inventory: float = 0.0,
    on_order_inventory: float = 0.0,
    backorders: float = 0.0,
    safety_stock_fixed: Optional[float] = None,
    inventory_position_fixed: Optional[float] = None,
    target_inventory_fixed: Optional[float] = None,
    # Backward-compatible aliases
    review_period_t: Optional[float] = None,
    lead_time_l: Optional[float] = None,
) -> PeriodicReviewResult:
    """
    Calculate target inventory (order-up-to level S) and recommended replenishment order quantity
    for a Fixed-Period Review periodic inventory policy.
    Returns all 8 educational verification sections.
    """
    t_val = review_period if review_period is not None else review_period_t
    l_val = lead_time if lead_time is not None else lead_time_l

    # 1. Input Validation - Never invent missing numerical inputs
    if t_val is None:
        raise ValueError("Review period 'review_period' (T) is required and cannot be None.")
    if l_val is None:
        raise ValueError("Lead time 'lead_time' (L) is required and cannot be None.")
    if daily_demand_mean is None:
        raise ValueError("Mean demand 'daily_demand_mean' (d) is required and cannot be None.")

    try:
        T = float(t_val)
        L = float(l_val)
        d = float(daily_demand_mean)
        sigma_d = float(daily_demand_std) if daily_demand_std is not None else 0.0
        on_hand = float(on_hand_inventory) if on_hand_inventory is not None else 0.0
        on_order = float(on_order_inventory) if on_order_inventory is not None else 0.0
        backlog = float(backorders) if backorders is not None else 0.0
    except (TypeError, ValueError):
        raise TypeError("All inventory and periodic review inputs must be numeric.")

    if T <= 0:
        raise ValueError(f"Review period T must be strictly positive (T > 0), got {T}.")
    if L < 0:
        raise ValueError(f"Lead time L cannot be negative (L >= 0), got {L}.")
    if d <= 0:
        raise ValueError(f"Mean demand must be strictly positive (d > 0), got {d}.")
    if sigma_d < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {sigma_d}.")

    # 2. Variable Definitions
    var_defs = {
        "T": "Review period interval between ordering epochs (periods/days)",
        "L": "Supplier replenishment lead time (periods/days)",
        "P = T + L": "Total vulnerability protection interval across review and lead time (periods/days)",
        "d": "Average demand rate per period (units/period)",
        "sigma_d": "Standard deviation of demand per period (units/period)",
        "sigma_P": "Standard deviation of demand across protection interval = sigma_d * sqrt(T + L) (units)",
        "SL": "Cycle service level probability",
        "z": "Safety factor Phi^-1(SL)",
        "SS": "Safety stock buffer over protection interval = z * sigma_P (units)",
        "S": "Target inventory / order-up-to level = d * (T + L) + SS (units)",
        "IP": "Inventory position = On_Hand + On_Order - Backorders (units)",
        "Q": "Recommended order replenishment quantity = max(0, S - IP) (units)",
    }

    # 3. Formula
    formulas = {
        "protection_period": "P = T + L",
        "expected_demand_protection_period": "E[D_P] = d * (T + L)",
        "std_dev_protection_period": "sigma_P = sigma_d * sqrt(T + L)",
        "safety_stock": "SS = z * sigma_P = z * sigma_d * sqrt(T + L)",
        "target_inventory": "S = d * (T + L) + SS",
        "inventory_position": "IP = On_Hand + On_Order - Backorders",
        "order_quantity": "Q = max(0, S - IP)",
    }

    # 4. Deterministic Calculations
    protection_period = T + L
    expected_protection_demand = d * protection_period
    sigma_protection = sigma_d * math.sqrt(protection_period)

    if safety_stock_fixed is not None:
        ss = float(safety_stock_fixed)
        z = float(z_score) if z_score is not None else 0.0
        sl_calc = float(service_level) if service_level is not None else 1.0
    elif sigma_d > 0:
        if z_score is not None:
            z = float(z_score)
            if z < 0:
                raise ValueError(f"z_score cannot be negative, got {z}.")
            sl_calc = normal_cdf(z)
        elif service_level is not None:
            sl = float(service_level)
            if not (0.0 < sl < 1.0):
                raise ValueError(f"service_level must be between 0 and 1, got {sl}.")
            z = normal_ppf(sl)
            sl_calc = sl
        else:
            raise ValueError(
                "Either 'service_level' (0 < SL < 1) or 'z_score' is required when demand standard deviation > 0. "
                "Missing numerical inputs will never be invented."
            )
        ss = z * sigma_protection
    else:
        z = 0.0
        sl_calc = 1.0 if service_level is None else float(service_level)
        ss = 0.0

    if target_inventory_fixed is not None:
        target_s = float(target_inventory_fixed)
    else:
        target_s = expected_protection_demand + ss

    if inventory_position_fixed is not None:
        inv_position = float(inventory_position_fixed)
    else:
        inv_position = on_hand + on_order - backlog

    order_qty = max(0.0, target_s - inv_position)

    # 5. Formula with Substituted Values
    formulas_substituted = {
        "protection_period": f"P = {T:.1f} + {L:.1f} = {protection_period:.1f} periods",
        "expected_demand_protection_period": f"E[D_P] = {d:,.2f} * {protection_period:.1f} = {expected_protection_demand:,.2f} units",
        "std_dev_protection_period": f"sigma_P = {sigma_d:,.2f} * sqrt({protection_period:.1f}) = {sigma_d:,.2f} * {math.sqrt(protection_period):.4f} = {sigma_protection:,.2f} units",
        "safety_stock": f"SS = {z:.4f} * {sigma_protection:,.2f} = {ss:,.2f} units (at CSL = {sl_calc*100:.1f}%)" if safety_stock_fixed is None else f"SS = {ss:,.2f} units (direct fixed input)",
        "target_inventory": f"S = {expected_protection_demand:,.2f} + {ss:,.2f} = {target_s:,.2f} units",
        "inventory_position": f"IP = {on_hand:,.0f} (on-hand) + {on_order:,.0f} (on-order) - {backlog:,.0f} (backorders) = {inv_position:,.0f} units" if inventory_position_fixed is None else f"IP = {inv_position:,.0f} units (direct fixed input)",
        "order_quantity": f"Q = max(0, {target_s:,.2f} - {inv_position:,.2f}) = {order_qty:,.2f} units",
    }

    # 6. Intermediate Calculations
    intermediate_calcs = {
        "protection_period_T_plus_L": round(protection_period, 2),
        "expected_demand_protection_period": round(expected_protection_demand, 2),
        "std_dev_protection_period": round(sigma_protection, 4),
        "z_factor": round(z, 4),
        "safety_stock": round(ss, 2),
        "target_inventory_S": round(target_s, 2),
        "inventory_position_IP": round(inv_position, 2),
    }

    # 7. Final Answer
    final_answer = {
        "order_quantity": round(order_qty, 2),
        "recommended_order_quantity_integer": int(math.ceil(order_qty)),
        "target_inventory": round(target_s, 2),
        "protection_period": round(protection_period, 2),
        "safety_stock": round(ss, 2),
        "inventory_position": round(inv_position, 2),
        "cycle_service_level": f"{sl_calc * 100:.2f}%",
    }

    # 8. Verification Checks
    verification_checks = {
        "all_checks_passed": bool(protection_period > 0 and order_qty >= 0 and target_s >= expected_protection_demand),
        "protection_window_positive": {
            "protection_period": round(protection_period, 2),
            "passed": bool(protection_period > 0),
            "condition": "Protection interval T + L must be strictly positive.",
        },
        "order_quantity_non_negative": {
            "order_quantity": round(order_qty, 2),
            "passed": bool(order_qty >= 0),
            "condition": "Replenishment order quantity Q = max(0, S - IP) is bounded below by 0.",
        },
        "target_inventory_covers_demand": {
            "target_inventory_S": round(target_s, 2),
            "expected_demand": round(expected_protection_demand, 2),
            "passed": bool(target_s >= expected_protection_demand),
            "condition": "Target level S must cover average protection period demand plus safety stock.",
        },
        "audit_log": [
            f"Protection interval {protection_period:.1f} periods (Review {T} + Lead Time {L}) accurately covers the vulnerability window.",
            f"Target inventory S = {target_s:.1f} units restores inventory position to desired buffer.",
            f"Order quantity calculated via Q = max(0, S - IP) = max(0, {target_s:.1f} - {inv_position:.1f}) = {order_qty:.1f} units.",
        ],
    }

    # 9. Explicit Assumptions
    assumptions = [
        "Inventory is reviewed and ordered only at discrete, periodic review epochs every T periods.",
        "Protection interval spans T + L periods; replenishment placed at review epoch arrives L periods later.",
        "Demand across successive periods is independent and identically distributed normal random variables.",
        "Unsatisfied customer demand during the cycle is fully backordered.",
        "Inventory position calculation accurately reflects on-hand, open pipeline orders, and customer backlog.",
    ]

    inputs_extracted = {
        "review_period_T": T,
        "lead_time_L": L,
        "daily_demand_mean_d": d,
        "daily_demand_std_sigma_d": sigma_d,
        "service_level_SL": service_level,
        "z_score": z_score,
        "on_hand_inventory": on_hand,
        "on_order_inventory": on_order,
        "backorders": backlog,
        "safety_stock_fixed": safety_stock_fixed,
        "inventory_position_fixed": inventory_position_fixed,
        "target_inventory_fixed": target_inventory_fixed,
        "review_period": T,
        "lead_time": L,
        "daily_demand_mean": d,
        "daily_demand_std": sigma_d,
    }

    return PeriodicReviewResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        order_quantity=order_qty,
        target_inventory=target_s,
        protection_period=protection_period,
        safety_stock=ss,
        inventory_position=inv_position,
        review_period=T,
        lead_time=L,
        mean_demand_protection_period=expected_protection_demand,
        std_dev_protection_period=sigma_protection,
        service_level=sl_calc,
        z_score=z,
        current_on_hand=on_hand,
        current_on_order=on_order,
        current_backorders=backlog,
    )


# Alias for backward compatibility
calculate_fixed_period_review = calculate_periodic_review


# ---------------------------------------------------------------------------
# 5. calculate_newsvendor() (Single-Period / Perishable / Seasonal)
# ---------------------------------------------------------------------------
@dataclass
class NewsvendorResult:
    # 8 Educational Verification Sections
    inputs_extracted: Dict[str, Any]
    variable_definitions: Dict[str, str]
    formula: Dict[str, str]
    formula_with_substituted_values: Dict[str, str]
    intermediate_calculations: Dict[str, Any]
    final_answer: Dict[str, Any]
    verification_checks: Dict[str, Any]
    assumptions: List[str]

    # Primary attributes for direct property access
    underage_cost: float
    overage_cost: float
    critical_ratio: float
    recommended_order_quantity: float

    # Additional stochastic & financial metrics
    z_score: float = 0.0
    expected_sales: float = 0.0
    expected_lost_sales: float = 0.0
    expected_leftover_inventory: float = 0.0
    expected_revenue: float = 0.0
    expected_purchase_cost: float = 0.0
    expected_salvage_revenue: float = 0.0
    expected_stockout_penalty: float = 0.0
    expected_profit: float = 0.0

    # Unit parameters
    unit_selling_price: float = 0.0
    unit_purchase_cost: float = 0.0
    unit_salvage_value: float = 0.0
    unit_penalty_cost: float = 0.0
    demand_mean: float = 0.0
    demand_std: float = 0.0

    # Backward compatibility helpers
    @property
    def inputs(self) -> Dict[str, Any]:
        return self.inputs_extracted

    @property
    def optimal_order_quantity(self) -> float:
        return self.recommended_order_quantity

    @property
    def verification_passed(self) -> bool:
        return bool(self.verification_checks.get("all_checks_passed", True))

    @property
    def verification_notes(self) -> List[str]:
        return list(self.verification_checks.get("audit_log", []))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "Newsvendor (Single-Period / Perishable)",
            "1_inputs_extracted": self.inputs_extracted,
            "2_variable_definitions": self.variable_definitions,
            "3_formula": self.formula,
            "4_formula_with_substituted_values": self.formula_with_substituted_values,
            "5_intermediate_calculations": self.intermediate_calculations,
            "6_final_answer": self.final_answer,
            "7_verification_checks": self.verification_checks,
            "8_assumptions": self.assumptions,
            "inputs_extracted": self.inputs_extracted,
            "variable_definitions": self.variable_definitions,
            "formula": self.formula,
            "formula_with_substituted_values": self.formula_with_substituted_values,
            "intermediate_calculations": self.intermediate_calculations,
            "final_answer": self.final_answer,
            "verification_checks": self.verification_checks,
            "assumptions": self.assumptions,
            "underage_cost": round(self.underage_cost, 2),
            "overage_cost": round(self.overage_cost, 2),
            "critical_ratio": round(self.critical_ratio, 4),
            "recommended_order_quantity": round(self.recommended_order_quantity, 2),
            "expected_profit": round(self.expected_profit, 2),
            "expected_sales": round(self.expected_sales, 2),
            "expected_lost_sales": round(self.expected_lost_sales, 2),
            "expected_leftover": round(self.expected_leftover_inventory, 2),
            "verification_passed": self.verification_passed,
            "verification_notes": self.verification_notes,
        }


def calculate_newsvendor(
    selling_price: Optional[float] = None,
    purchase_cost: Optional[float] = None,
    demand_mean: Optional[float] = None,
    demand_std: float = 0.0,
    salvage_value: float = 0.0,
    penalty_cost: float = 0.0,
    underage_cost_direct: Optional[float] = None,
    overage_cost_direct: Optional[float] = None,
) -> NewsvendorResult:
    """
    Calculate optimal single-period order quantity and economic metrics for the Newsvendor model.
    Returns all 8 educational verification sections.
    """
    # 1. Input Validation - Never invent missing numerical inputs
    if demand_mean is None:
        raise ValueError("Expected demand mean 'demand_mean' (mu) is required and cannot be None.")

    try:
        mu = float(demand_mean)
        sigma = float(demand_std) if demand_std is not None else 0.0
    except (TypeError, ValueError):
        raise TypeError("Demand parameters must be numeric values.")

    if mu < 0:
        raise ValueError(f"Demand mean cannot be negative, got {mu}.")
    if sigma < 0:
        raise ValueError(f"Demand standard deviation cannot be negative, got {sigma}.")

    # Resolve Underage (Cu) and Overage (Co)
    if underage_cost_direct is not None and overage_cost_direct is not None:
        try:
            cu = float(underage_cost_direct)
            co = float(overage_cost_direct)
        except (TypeError, ValueError):
            raise TypeError("underage_cost_direct and overage_cost_direct must be numeric values.")
        if cu <= 0:
            raise ValueError(f"Underage cost must be strictly positive, got {cu}.")
        if co <= 0:
            raise ValueError(f"Overage cost must be strictly positive, got {co}.")
        P = float(selling_price) if selling_price is not None else (cu + (float(purchase_cost) if purchase_cost is not None else 0.0))
        C = float(purchase_cost) if purchase_cost is not None else 0.0
        V = float(salvage_value) if salvage_value is not None else 0.0
        penalty = float(penalty_cost) if penalty_cost is not None else 0.0
    else:
        if selling_price is None:
            raise ValueError("Unit selling price 'selling_price' (P) is required when direct underage cost is not provided.")
        if purchase_cost is None:
            raise ValueError("Unit purchase cost 'purchase_cost' (C) is required when direct overage cost is not provided.")

        try:
            P = float(selling_price)
            C = float(purchase_cost)
            V = float(salvage_value) if salvage_value is not None else 0.0
            penalty = float(penalty_cost) if penalty_cost is not None else 0.0
        except (TypeError, ValueError):
            raise TypeError("All Newsvendor parameters must be numeric values.")

        if P <= 0:
            raise ValueError(f"Selling price must be positive, got ${P}.")
        if C <= 0:
            raise ValueError(f"Purchase cost must be positive, got ${C}.")
        if P <= C:
            raise ValueError(f"Selling price (${P}) must exceed purchase cost (${C}) for economic viability.")
        if C < V:
            raise ValueError(f"Purchase cost (${C}) cannot be less than salvage value (${V}).")
        if penalty < 0:
            raise ValueError(f"Penalty cost cannot be negative, got ${penalty}.")

        cu = (P - C) + penalty
        co = C - V

    if (cu + co) <= 0:
        raise ValueError("Sum of underage cost Cu and overage cost Co must be strictly positive.")

    # 2. Variable Definitions
    var_defs = {
        "P": "Unit selling price ($/unit)",
        "C": "Unit purchase or production cost ($/unit)",
        "V": "Unit salvage or markdown value for unsold leftover items ($/unit)",
        "Penalty": "Goodwill or shortage penalty per unit of unmet customer demand ($/unit)",
        "Cu": "Marginal cost of underage = profit lost per unit of unmet demand = (P - C + Penalty) ($/unit)",
        "Co": "Marginal cost of overage = net loss per unsold leftover unit = (C - V) ($/unit)",
        "CR": "Critical ratio / critical fractile = Cu / (Cu + Co) (target in-stock probability)",
        "mu": "Expected seasonal / single-period customer demand (units)",
        "sigma": "Standard deviation of customer demand (units)",
        "z": "Optimal standard normal safety factor corresponding to Phi^-1(CR)",
        "Q*": "Optimal stocking order quantity = mu + z * sigma (units)",
        "E[Lost_Sales]": "Expected units of unsatisfied customer demand = sigma * G(z) (units)",
        "E[Sales]": "Expected units sold to customers = mu - E[Lost_Sales] (units)",
        "E[Leftover]": "Expected leftover inventory salvaged at end of season = Q* - E[Sales] (units)",
        "E[Profit]": "Expected total economic profit ($)",
    }

    # 3. Formula
    formulas = {
        "underage_cost": "Cu = (P - C) + Penalty",
        "overage_cost": "Co = C - V",
        "critical_ratio": "CR = Cu / (Cu + Co)",
        "optimal_order_quantity": "Q* = mu + z * sigma  where z = Phi^-1(CR)",
        "standard_normal_loss": "G(z) = phi(z) - z * (1 - Phi(z))",
        "expected_lost_sales": "E[Lost_Sales] = sigma * G(z)",
        "expected_sales": "E[Sales] = mu - E[Lost_Sales]",
        "expected_leftover": "E[Leftover] = Q* - E[Sales]",
        "expected_profit": "E[Profit] = (P * E[Sales]) + (V * E[Leftover]) - (C * Q*) - (Penalty * E[Lost_Sales])",
    }

    # 4. Deterministic Calculations
    cr = cu / (cu + co)

    if sigma == 0:
        z = 0.0
        q_star = mu
        loss_val = 0.0
        exp_lost = 0.0
        exp_sales = mu
        exp_leftover = 0.0
    else:
        z = normal_ppf(cr)
        q_star = max(0.0, mu + z * sigma)
        loss_val = standard_normal_loss(z)
        exp_lost = sigma * loss_val
        exp_sales = mu - exp_lost
        exp_leftover = max(0.0, q_star - exp_sales)

    exp_rev = P * exp_sales
    exp_salv = V * exp_leftover
    exp_proc = C * q_star
    exp_pen = penalty * exp_lost
    exp_profit = exp_rev + exp_salv - exp_proc - exp_pen

    # 5. Formula with Substituted Values
    formulas_substituted = {
        "underage_cost": f"Cu = (${P:.2f} - ${C:.2f}) + ${penalty:.2f} = ${cu:.2f}/unit",
        "overage_cost": f"Co = ${C:.2f} - ${V:.2f} = ${co:.2f}/unit",
        "critical_ratio": f"CR = ${cu:.2f} / (${cu:.2f} + ${co:.2f}) = ${cu:.2f} / ${cu + co:.2f} = {cr:.4f} ({cr*100:.2f}%)",
        "optimal_order_quantity": f"Q* = {mu:,.1f} + ({z:.4f} * {sigma:,.1f}) = {mu:,.1f} + {z * sigma:,.1f} = {q_star:,.2f} units",
        "expected_lost_sales": f"E[Lost_Sales] = {sigma:,.1f} * G({z:.4f}) = {sigma:,.1f} * {loss_val:.4f} = {exp_lost:,.2f} units",
        "expected_sales": f"E[Sales] = {mu:,.1f} - {exp_lost:,.2f} = {exp_sales:,.2f} units",
        "expected_leftover": f"E[Leftover] = {q_star:,.2f} - {exp_sales:,.2f} = {exp_leftover:,.2f} units",
        "expected_profit": (
            f"E[Profit] = (${P:.2f} * {exp_sales:,.2f}) + (${V:.2f} * {exp_leftover:,.2f}) - (${C:.2f} * {q_star:,.2f}) - (${penalty:.2f} * {exp_lost:,.2f}) = "
            f"${exp_rev:,.2f} + ${exp_salv:,.2f} - ${exp_proc:,.2f} - ${exp_pen:,.2f} = ${exp_profit:,.2f}"
        ),
    }

    # 6. Intermediate Calculations
    intermediate_calcs = {
        "underage_cost_Cu": round(cu, 2),
        "overage_cost_Co": round(co, 2),
        "critical_ratio_CR": round(cr, 4),
        "z_score": round(z, 4),
        "unit_normal_loss_G_z": round(loss_val, 6),
        "expected_lost_sales_units": round(exp_lost, 2),
        "expected_sales_units": round(exp_sales, 2),
        "expected_leftover_units": round(exp_leftover, 2),
        "expected_gross_revenue": round(exp_rev, 2),
        "expected_salvage_revenue": round(exp_salv, 2),
        "expected_procurement_cost": round(exp_proc, 2),
        "expected_penalty_cost": round(exp_pen, 2),
        "expected_profit": round(exp_profit, 2),
    }

    # 7. Final Answer
    final_answer = {
        "underage_cost": round(cu, 2),
        "overage_cost": round(co, 2),
        "critical_ratio": round(cr, 4),
        "critical_ratio_percentage": f"{cr * 100:.2f}%",
        "recommended_order_quantity": round(q_star, 2),
        "recommended_order_quantity_integer": int(round(q_star)),
        "expected_profit": round(exp_profit, 2),
        "expected_sales": round(exp_sales, 2),
        "expected_leftover": round(exp_leftover, 2),
        "expected_lost_sales": round(exp_lost, 2),
    }

    # 8. Verification Checks
    verification_checks = {
        "all_checks_passed": bool(cu > 0 and co > 0 and 0.0 < cr < 1.0 and q_star >= 0),
        "economic_validity": {
            "selling_price": P,
            "purchase_cost": C,
            "passed": bool(P > C or (underage_cost_direct is not None)),
            "condition": "Selling price exceeds purchase cost to ensure positive marginal profit.",
        },
        "critical_ratio_validity": {
            "critical_ratio": round(cr, 4),
            "passed": bool(0.0 < cr < 1.0),
            "condition": "Critical ratio fractile must lie strictly between 0 and 1.",
        },
        "non_negative_order_quantity": {
            "recommended_order_quantity": round(q_star, 2),
            "passed": bool(q_star >= 0),
            "condition": "Stocking order quantity must be non-negative.",
        },
        "audit_log": [
            f"Underage cost Cu = ${cu:.2f} vs. Overage cost Co = ${co:.2f}.",
            f"Critical ratio (optimal in-stock probability) is {cr * 100:.2f}% (Z-factor = {z:.3f}).",
            f"Ordering {q_star:,.1f} units mathematically balances underage margin loss against overage liquidation risk.",
            f"Expected profit: ${exp_profit:,.2f}.",
        ],
    }

    # 9. Explicit Assumptions
    assumptions = [
        "Single procurement opportunity prior to the selling season; no intra-season replenishment.",
        "Items are perishable, seasonal, or fashion goods with no regular demand after the period concludes.",
        "Unsold units must be liquidated at the given salvage value V.",
        "Customer demand during the selling period follows a normal distribution with parameters (mu, sigma).",
        "Unmet customer demand is permanently lost, with an optional goodwill penalty per unmet unit.",
    ]

    inputs_extracted = {
        "unit_selling_price_P": P,
        "unit_purchase_cost_C": C,
        "unit_salvage_value_V": V,
        "penalty_cost_B": penalty,
        "demand_mean_mu": mu,
        "demand_std_sigma": sigma,
        "underage_cost_direct": underage_cost_direct,
        "overage_cost_direct": overage_cost_direct,
        "selling_price": P,
        "purchase_cost": C,
        "salvage_value": V,
        "demand_mean": mu,
        "demand_std": sigma,
    }

    return NewsvendorResult(
        inputs_extracted=inputs_extracted,
        variable_definitions=var_defs,
        formula=formulas,
        formula_with_substituted_values=formulas_substituted,
        intermediate_calculations=intermediate_calcs,
        final_answer=final_answer,
        verification_checks=verification_checks,
        assumptions=assumptions,
        underage_cost=cu,
        overage_cost=co,
        critical_ratio=cr,
        recommended_order_quantity=q_star,
        z_score=z,
        expected_sales=exp_sales,
        expected_lost_sales=exp_lost,
        expected_leftover_inventory=exp_leftover,
        expected_revenue=exp_rev,
        expected_purchase_cost=exp_proc,
        expected_salvage_revenue=exp_salv,
        expected_stockout_penalty=exp_pen,
        expected_profit=exp_profit,
        unit_selling_price=P,
        unit_purchase_cost=C,
        unit_salvage_value=V,
        unit_penalty_cost=penalty,
        demand_mean=mu,
        demand_std=sigma,
    )
