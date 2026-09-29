"""Deterministic LABELED legibility corpus for measuring the gate.

Every corrupted chunk is DERIVED from a clean one by the same real, mechanical
fault transforms used in fixtures.py (interleave two columns; flatten table
delimiters). Nothing is hand-labeled as "bad": the label follows from the
transform, and the gate must recover it from the text alone.

The corpus includes these HARD cases:
  * same-topic interleave (columns share vocabulary, so the order margin is
    much smaller than the topically-distinct case),
  * uniform / boilerplate legal prose (a false-positive risk for the order
    signal, ironic given the 'legi-' name),
  * several numeric tables (clean + collapsed).

label == 1  => CORRUPTED, the gate SHOULD quarantine it.
label == 0  => CLEAN, the gate SHOULD keep it.
"""
import re

from fixtures import COLUMN_A, COLUMN_B, CLEAN_TABLE

# --- extra clean single-column prose passages -------------------------------
# Each is locally coherent: consecutive lines share a keyword (the property the
# reading-order signal relies on). Topics vary, incl. uniform legal boilerplate.

RIVERS = [
    "A river begins as small streams high in the mountains.",
    "These mountain streams merge into a single flowing channel.",
    "The flowing channel carves a valley through soft rock.",
    "Soft rock erodes faster than the harder stone beside it.",
    "Beside the harder stone the water slows and drops silt.",
    "Dropped silt builds fertile plains along the lower river.",
    "The lower river finally widens into a broad calm delta.",
    "A broad delta spills the river gently into the open sea.",
]

BREAD = [
    "Baking bread starts with mixing flour water and yeast.",
    "The yeast feeds on sugars in the flour and makes gas.",
    "Gas bubbles stretch the dough and make it slowly rise.",
    "As it rises the dough doubles in size over an hour.",
    "After an hour the dough is shaped into a round loaf.",
    "The round loaf rests again before it goes in the oven.",
    "In the hot oven the loaf sets into a firm brown crust.",
    "A firm brown crust seals the soft crumb inside the bread.",
]

LEGAL = [
    "This agreement is entered into by and between the parties.",
    "The parties agree to the terms set forth in this agreement.",
    "Each term set forth herein shall bind both of the parties.",
    "Both parties acknowledge that they have read each term.",
    "Having read each term the parties accept the obligations.",
    "The obligations of the parties continue for the full term.",
    "For the full term neither party may assign this agreement.",
    "This agreement supersedes all prior agreements between them.",
]

COFFEE = [
    "Coffee begins as a red cherry growing on a green shrub.",
    "The green shrub thrives in cool high tropical highlands.",
    "In the highlands workers pick each ripe cherry by hand.",
    "Picked cherries are pulped to free the pale inner beans.",
    "The pale beans are dried slowly under the warm morning sun.",
    "Dried beans are roasted until they turn dark and fragrant.",
    "Dark fragrant beans are ground just before they are brewed.",
    "Brewing the ground beans with hot water yields a fresh cup.",
]

BEES = [
    "A honeybee colony lives inside a carefully built hive.",
    "Inside the hive thousands of workers share every task.",
    "Worker bees fly out to gather nectar from open flowers.",
    "Nectar from the flowers is carried back to the waiting hive.",
    "Back at the hive the nectar is turned slowly into honey.",
    "Ripe honey is stored in neat wax cells for the cold winter.",
    "Through the cold winter the stored honey feeds the colony.",
    "The colony survives until warm spring flowers bloom again.",
]

# Second same-topic passages, for the HARD (same-topic) interleave pairs.
PHOTO_2 = [
    "Sunlight is the primary energy source for green plants.",
    "Green plants trap that sunlight inside tiny leaf cells.",
    "Each leaf cell holds green pigment packed in chloroplasts.",
    "Chloroplasts use the pigment to capture incoming photons.",
    "Captured photons power reactions that build plant sugars.",
    "Plant sugars store the captured energy as chemical bonds.",
    "These chemical bonds fuel growth throughout the whole plant.",
    "The whole plant depends on sunlight captured by its leaves.",
]

REVENUE_2 = [
    "The finance team reviewed revenue across every division.",
    "Every division reported its sales for the fiscal quarter.",
    "Fiscal quarter sales were compiled into a single report.",
    "The single report showed earnings above the prior year.",
    "Prior year earnings had lagged behind the market average.",
    "The market average was pulled down by weak supply chains.",
    "Weak supply chains raised costs across the whole sector.",
    "The whole sector now watches margins for the coming year.",
]

# --- extra clean numeric tables ---------------------------------------------
TABLE_PRODUCTS = [
    "Product     Price    Stock",
    "Widget       12.50      340",
    "Gadget       27.00      120",
    "Gizmo         9.75      560",
    "Sprocket     18.20      210",
]

TABLE_YEARS = [
    "Year    Sales    Profit",
    "2019     4200      310",
    "2020     3800      120",
    "2021     5100      640",
    "2022     6700      910",
]

TABLE_CITIES = [
    "City       Temp    Rain",
    "Denver       31      12",
    "Miami        84      55",
    "Boston       45      38",
    "Seattle      52      41",
]


# --- fault-injection transforms (identical mechanics to fixtures.py) --------
def _interleave(a, b):
    """Two columns interleaved line-by-line (A1,B1,A2,B2,...)."""
    rows = []
    for x, y in zip(a, b):
        rows.append(x)
        rows.append(y)
    return "\n".join(rows)


def _collapse(table_lines):
    """Flatten every 2+ space run to a single space, so cell structure is lost."""
    return "\n".join(re.sub(r" {2,}", " ", ln).strip() for ln in table_lines)


def _join(lines):
    return "\n".join(lines)


# --- the labeled corpus ------------------------------------------------------
def labeled_corpus():
    """Return list of (chunk_id, text, label). label 1 = corrupted, 0 = clean."""
    items = []

    # ---- CLEAN prose (label 0) ----
    clean_prose = {
        "clean_photosynthesis": COLUMN_A,
        "clean_photosynthesis2": PHOTO_2,
        "clean_revenue": COLUMN_B,
        "clean_revenue2": REVENUE_2,
        "clean_rivers": RIVERS,
        "clean_bread": BREAD,
        "clean_legal_boilerplate": LEGAL,   # uniform prose, a chosen FP risk
        "clean_coffee": COFFEE,
        "clean_bees": BEES,
    }
    for cid, lines in clean_prose.items():
        items.append((cid, _join(lines), 0))

    # ---- CLEAN tables (label 0) ----
    clean_tables = {
        "clean_table_regions": CLEAN_TABLE,
        "clean_table_products": TABLE_PRODUCTS,
        "clean_table_years": TABLE_YEARS,
        "clean_table_cities": TABLE_CITIES,
    }
    for cid, lines in clean_tables.items():
        items.append((cid, _join(lines), 0))

    # ---- CORRUPTED: cross-topic interleave (easy) (label 1) ----
    items.append(("corrupt_interleave_photo_x_revenue",
                  _interleave(COLUMN_A, COLUMN_B), 1))
    items.append(("corrupt_interleave_rivers_x_bread",
                  _interleave(RIVERS, BREAD), 1))
    items.append(("corrupt_interleave_coffee_x_bees",
                  _interleave(COFFEE, BEES), 1))
    items.append(("corrupt_interleave_legal_x_coffee",
                  _interleave(LEGAL, COFFEE), 1))

    # ---- CORRUPTED: same-topic interleave (HARD: shared vocabulary) ----
    items.append(("corrupt_interleave_photo_SAME",
                  _interleave(COLUMN_A, PHOTO_2), 1))
    items.append(("corrupt_interleave_revenue_SAME",
                  _interleave(COLUMN_B, REVENUE_2), 1))

    # ---- CORRUPTED: collapsed tables (label 1) ----
    items.append(("corrupt_collapse_regions", _collapse(CLEAN_TABLE), 1))
    items.append(("corrupt_collapse_products", _collapse(TABLE_PRODUCTS), 1))
    items.append(("corrupt_collapse_years", _collapse(TABLE_YEARS), 1))
    items.append(("corrupt_collapse_cities", _collapse(TABLE_CITIES), 1))

    return items
