from __future__ import annotations

import pytest


def test_no_piranha_plant_in_catalog(config, catalog) -> None:
    """Verify no piranha plant in catalog."""
    with pytest.raises(KeyError):
        catalog.id_for_name("piranha_plant")
