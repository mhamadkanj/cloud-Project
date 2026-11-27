"""Event-driven pipeline: three functions in one Function App (Flex Consumption, Python 3.12).

dispatcher  Event Grid (BlobCreated on input/) -> validate -> queue message, or move to rejected/
worker      Queue trigger (inference-requests) -> call the API -> output/ JSON + Cosmos DB upsert
            5 failed attempts -> message moved to inference-requests-poison (host.json)
recent      HTTP GET /api/recent -> last forecasts from Cosmos DB, for the dashboard

Authentication: the Function App's managed identity (DefaultAzureCredential).
No key or connection string in the code; settings come from app settings:
  STORAGE_ACCOUNT_NAME, API_URL, COSMOS_ENDPOINT, COSMOS_DATABASE, COSMOS_CONTAINER,
  DataStorage__queueServiceUri (identity-based connection for the queue trigger)
"""
import json
import logging
import os
from functools import lru_cache

import azure.functions as func
import requests
from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue import QueueClient, TextBase64EncodePolicy

from pipeline import load_actuals, make_item, validate_csv

INPUT, OUTPUT, REJECTED = "input", "output", "rejected"
QUEUE_NAME = "inference-requests"

app = func.FunctionApp()


# ---------- clients, created once per instance ----------
@lru_cache
def credential():
    return DefaultAzureCredential()


@lru_cache
def blobs() -> BlobServiceClient:
    account = os.environ["STORAGE_ACCOUNT_NAME"]
    return BlobServiceClient(f"https://{account}.blob.core.windows.net", credential=credential())


@lru_cache
def queue() -> QueueClient:
    account = os.environ["STORAGE_ACCOUNT_NAME"]
    # the queue trigger expects base64 messages (Functions default)
    return QueueClient(f"https://{account}.queue.core.windows.net", QUEUE_NAME,
                       credential=credential(), message_encode_policy=TextBase64EncodePolicy())


@lru_cache
def inferences():
    client = CosmosClient(os.environ["COSMOS_ENDPOINT"], credential=credential())
    db = client.get_database_client(os.environ.get("COSMOS_DATABASE", "pipeline"))
    return db.get_container_client(os.environ.get("COSMOS_CONTAINER", "inferences"))


@lru_cache
def actuals():
    return load_actuals()


# ---------- 1. dispatcher ----------
@app.function_name("dispatcher")
@app.event_grid_trigger(arg_name="event")
def dispatcher(event: func.EventGridEvent) -> None:
    subject = event.subject  # /blobServices/default/containers/input/blobs/<name>
    if "/containers/input/blobs/" not in subject:
        logging.info("dispatcher: ignored event %s", subject)
        return
    name = subject.split("/blobs/", 1)[1]
    source = blobs().get_blob_client(INPUT, name)
    etag = source.get_blob_properties().etag.strip('"')
    content = source.download_blob().readall()

    rows, reason = validate_csv(name, content)
    if rows is None:
        target = blobs().get_blob_client(REJECTED, name)
        safe_reason = reason.encode("ascii", "replace").decode()[:250]
        target.upload_blob(content, overwrite=True, metadata={"reason": safe_reason})
        source.delete_blob()
        logging.warning("dispatcher: rejected %s (%s)", name, reason)
        return

    queue().send_message(json.dumps({"blob": name, "etag": etag}))
    logging.info("dispatcher: queued %s (etag %s)", name, etag)


# ---------- 2. worker ----------
@app.function_name("worker")
@app.queue_trigger(arg_name="msg", queue_name=QUEUE_NAME, connection="DataStorage")
def worker(msg: func.QueueMessage) -> None:
    job = json.loads(msg.get_body().decode("utf-8"))
    name, etag = job["blob"], job["etag"]
    attempt = msg.dequeue_count or 1
    logging.info("worker: %s attempt %s", name, attempt)

    content = blobs().get_blob_client(INPUT, name).download_blob().readall()
    rows, reason = validate_csv(name, content)
    if rows is None:  # file changed after dispatch: fail -> retried, then poison
        raise ValueError(f"{name} is no longer valid: {reason}")

    # any error here raises -> the message is retried; after 5 attempts -> poison queue
    api = os.environ["API_URL"].rstrip("/")
    response = requests.post(f"{api}/predict", json={"rows": rows, "job_hours": 2}, timeout=90)
    response.raise_for_status()
    result = response.json()

    item = make_item(name, etag, rows, result, attempt, actuals())
    out_name = name.rsplit(".", 1)[0] + ".json"
    blobs().get_blob_client(OUTPUT, out_name).upload_blob(
        json.dumps({"summary": item, "api_response": result}, indent=2),
        overwrite=True, content_settings=ContentSettings(content_type="application/json"))
    inferences().upsert_item(item)  # same id on retry -> no duplicate (idempotent)
    logging.info("worker: %s done, best window %s, saved %s%%",
                 name, item["best_window_start"], item["co2_saved_pct"])


# ---------- 3. recent ----------
@app.function_name("recent")
@app.route(route="recent", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def recent(req: func.HttpRequest) -> func.HttpResponse:
    try:
        n = max(1, min(int(req.params.get("n", "20")), 50))
    except ValueError:
        n = 20
    items = list(inferences().query_items(
        query="SELECT TOP @n * FROM c ORDER BY c.processed_at DESC",
        parameters=[{"name": "@n", "value": n}],
        enable_cross_partition_query=True))
    for item in items:  # drop Cosmos system fields
        for key in [k for k in item if k.startswith("_")]:
            item.pop(key)
    return func.HttpResponse(json.dumps(items), mimetype="application/json")
