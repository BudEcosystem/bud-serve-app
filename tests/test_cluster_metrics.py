import asyncio
import json
from uuid import UUID
from budapp.cluster_ops.utils import ClusterMetricsFetcher

TEST_CLUSTER_ID = UUID("dff4578f-039e-4be4-aa36-db7932e404fa")

async def test_get_cluster_metrics():
    fetcher = ClusterMetricsFetcher("https://metrics.fmops.in")
    metrics = await fetcher.get_cluster_metrics(TEST_CLUSTER_ID, "10min", "all")
    # write metrics to file
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    asyncio.run(test_get_cluster_metrics())
