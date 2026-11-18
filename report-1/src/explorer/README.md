
# Local running (inc. for development)

uv run python src/explorer/app.py [--port 8000]
then open http://localhost:8000



# New deployment (optional, based on GCP)

The explorer is ready to deploy to Cloud Run behind IAP

Dockerfile: installs only the packages the explorer needs. The data files are copied into it so it's self-contained.

The image is c.442 MB. Local tests use c.120 MB of memory.

Deploying: run `gcloud auth login`, then:

* Enable the Cloud Run, Cloud Build, Artifact Registry and IAP services in your GCP project.
* Build and deploy the service with IAP on and no public access. This prints the service URL.
* Give IAP's service account permission to call the service.


Consent screen: if IAP asks for an OAuth consent screen, create one once in the console and set it to Internal.

Speed: the first visit after a quiet spell will take a few seconds while the data loads, because the service scales down to zero when idle to keep costs near nothing.