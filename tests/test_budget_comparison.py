"""Invented fixtures only; no research outputs."""
import copy
import json
import subprocess
import sys
import unittest
from groupbench.budget_comparison import compare_saved_budgets, load_document


def fixture():
    doc = {"schema_version": 1, "methods": ["baseline", "selector"], "budgets": ["small", "large"],
           "metadata": {"metric_definition": "mean absolute error", "value_unit": "invented_units", "independent_unit_definition": "invented independent experiment", "normalization": "none; same scale in every cell", "direction": "minimize"}, "rows": []}
    # Unit A has 3 repeats, B has 1: equal unit, not equal repeat weighting.
    for unit, repeats, low, high in [("A", 3, 2, -2), ("B", 1, -2, 4), ("C", 1, 0, 0)]:
        for repeat in range(repeats):
            for budget, gain in [("small", low), ("large", high)]:
                for method, value in [("baseline", 10), ("selector", 10-gain)]:
                    doc["rows"].append(dict(unit_id=unit, repeat_id=str(repeat), budget=budget, method=method, value=value, status="ok", metric_definition="mean absolute error", value_unit="invented_units"))
    return doc


class BudgetComparisonTests(unittest.TestCase):
    def test_equal_unit_weight_and_interaction(self):
        report = compare_saved_budgets(fixture())
        self.assertEqual(report["by_budget"]["small"], dict(mean_unit_gain=0, strict_wins=1, ties=1, harms=1))
        self.assertAlmostEqual(report["by_budget"]["large"]["mean_unit_gain"], 2/3)
        self.assertAlmostEqual(report["mean_selector_by_budget_interaction"], 2/3)
        self.assertEqual(len(report["observations"]), 5)
        self.assertFalse(report["provenance_verified"])
        self.assertEqual(report["coverage"]["omitted_cells"], 0)

    def test_direction_and_order(self):
        doc = fixture(); original = compare_saved_budgets(doc)
        doc["metadata"]["direction"] = "maximize"
        self.assertEqual(compare_saved_budgets(doc)["mean_selector_by_budget_interaction"], -original["mean_selector_by_budget_interaction"])
        doc["methods"].reverse()
        self.assertEqual(compare_saved_budgets(doc)["mean_selector_by_budget_interaction"], original["mean_selector_by_budget_interaction"])
        doc["budgets"].reverse()
        self.assertEqual(compare_saved_budgets(doc)["mean_selector_by_budget_interaction"], -original["mean_selector_by_budget_interaction"])

    def test_complete_adverse_ledger(self):
        doc=fixture();doc["rows"][1]["value"]=20
        report=compare_saved_budgets(doc)
        self.assertEqual(report["observations"][0]["outcomes"]["small"], "harm")
        self.assertEqual(report["units"][0]["outcomes"]["small"], "harm")
        self.assertEqual(len(report["observations"])*4,len(doc["rows"]))

    def test_reject_invalid_cells(self):
        for key, value in [("value",float('nan')), ("value",float('inf')), ("value",True), ("value","1"), ("value",None), ("status","failed"), ("value_unit","other"), ("metric_definition","RMSE"), ("unit_id"," "), ("repeat_id",None), ("method","unknown"), ("budget","unknown")]:
            with self.subTest(key=key,value=value):
                doc=fixture();doc["rows"][0][key]=value
                with self.assertRaises(ValueError):compare_saved_budgets(doc)

    def test_missing_and_duplicate(self):
        for change in [lambda rows: rows.pop(),lambda rows: rows.append(rows[0].copy())]:
            doc=fixture();change(doc["rows"])
            with self.assertRaises(ValueError):compare_saved_budgets(doc)

    def test_mismatched_unit_repeat_sets(self):
        for key in ['unit_id','repeat_id']:
            doc=fixture();doc['rows'][0][key]='new'
            with self.assertRaises(ValueError):compare_saved_budgets(doc)

    def test_schema_fail_closed(self):
        for key,value in [('methods',['x','x']),('budgets',['x']),('rows',[]),('schema_version',True),('metadata',{})]:
            doc=fixture();doc[key]=value
            with self.assertRaises(ValueError):compare_saved_budgets(doc)
        doc=fixture();doc['uncertainty']=True
        with self.assertRaises(ValueError):compare_saved_budgets(doc)

    def test_json_duplicate_and_nonfinite(self):
        for text in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}']:
            with self.assertRaises(ValueError):load_document(text)

    def test_excessive_json_nesting_rejected(self):
        with self.assertRaisesRegex(ValueError, "nesting"):
            load_document("[" * 20000 + "0" + "]" * 20000)

    def test_exact_ties_and_small_strict_gain(self):
        doc=fixture();doc['rows'][1]['value']=9.999999999999998
        report=compare_saved_budgets(doc)
        self.assertEqual(report['observations'][0]['outcomes']['small'],'win')
        self.assertGreater(report['observations'][0]['method_b_gain']['small'],0)

    def test_numeric_range_reject(self):
        for token in ['1e999999999','1e-999999999']:
            doc=fixture();text=json.dumps(doc).replace('"value": 10','"value": '+token,1)
            with self.assertRaises(ValueError):compare_saved_budgets(load_document(text))

    def test_order_invariance(self):
        doc=fixture();report=compare_saved_budgets(doc);doc['rows'].reverse()
        self.assertEqual(report,compare_saved_budgets(doc))

if __name__ == '__main__':unittest.main()
