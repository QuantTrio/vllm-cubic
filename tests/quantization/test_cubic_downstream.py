# SPDX-License-Identifier: Apache-2.0

from tools.check_cubic_downstream import check_contracts


def test_downstream_contract_survives_upstream_sync() -> None:
    assert check_contracts() == []
