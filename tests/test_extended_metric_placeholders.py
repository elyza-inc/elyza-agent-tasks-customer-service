from elyza_agent_tasks_customer_service.evaluation.scoring.extended_metric_scoring import (
    score_extended_metrics,
)


def test_interaction_metrics_are_contract_invalid_placeholders() -> None:
    instances = [
        {
            "metric_instance_id": f"instance-{metric_id}",
            "metric_id": metric_id,
            "contract": {},
        }
        for metric_id in ("M16", "M19")
    ]

    report = score_extended_metrics(
        scenario={
            "scenario_id": "scenario",
            "measurement_contract": {"metric_instances": instances},
        },
        artifacts={},
    )

    assert [row["metric_id"] for row in report["metric_results"]] == ["M16", "M19"]
    assert all(row["status"] == "contract_invalid" for row in report["metric_results"])
    assert all(row["value"] is None for row in report["metric_results"])
