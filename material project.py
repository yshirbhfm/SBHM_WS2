# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:52:28 2022

@author: Lab KYLO
"""

from mp_api.client import MPRester

with MPRester("nO9yLtOZjeEHRDHPiiCFPkqgtqrPuvN5") as mpr:

    docs = mpr.summary.search(material_ids=["mp-149"], fields=["structure"])
    structure = docs[0].structure
    
    # -- Shortcut for a single Materials Project ID:
    structure = mpr.get_structure_by_material_id("mp-149")
    # a = structure._lattice.get_cartesian_coords([0.25,0.75,0.25])
    a = structure._sites[0]._lattice.get_cartesian_coords(0.25)