"""Optimizer ingredients (ADR 0050 / 0051).

The SOAP / ESOAP / SinkSOAPH / Muon / cautious-WD optimizer implementations
and their parameter-group routing, relocated here from ``lamquant/student/``
so every trainer selects an optimizer through one home instead of duplicated
``if/elif`` chains. The typed ``IngredientSpec`` registry that wraps these
lands in Phase 3 of the cookbook rebuild.
"""
