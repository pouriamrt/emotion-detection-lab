# ── Configuration ─────────────────────────────────────────────────────
PROJECT_ID   ?= $(shell gcloud config get-value project 2>/dev/null)
REGION       ?= us-central1
SERVICE_NAME ?= emotion-detection
IMAGE        ?= gcr.io/$(PROJECT_ID)/$(SERVICE_NAME)
MEMORY       ?= 4Gi
CPU          ?= 2
TIMEOUT      ?= 300

# ── Setup ─────────────────────────────────────────────────────────────
.PHONY: setup enable-apis create-secrets

setup: enable-apis create-secrets ## One-time GCP setup (APIs + secrets)

enable-apis: ## Enable required GCP APIs
	gcloud services enable \
		run.googleapis.com \
		cloudbuild.googleapis.com \
		secretmanager.googleapis.com \
		artifactregistry.googleapis.com

create-secrets: ## Upload secrets to GCP Secret Manager
	gcloud secrets create client-secret-json \
		--data-file=client_secret.json 2>/dev/null || \
		gcloud secrets versions add client-secret-json \
		--data-file=client_secret.json
	gcloud secrets create streamlit-secrets \
		--data-file=.streamlit/secrets.toml 2>/dev/null || \
		gcloud secrets versions add streamlit-secrets \
		--data-file=.streamlit/secrets.toml

update-secrets: ## Update existing secrets with current local files
	gcloud secrets versions add client-secret-json \
		--data-file=client_secret.json
	gcloud secrets versions add streamlit-secrets \
		--data-file=.streamlit/secrets.toml

# ── Build & Deploy ────────────────────────────────────────────────────
.PHONY: build deploy deploy-full

build: ## Build container image with Cloud Build
	gcloud builds submit --tag $(IMAGE)

deploy: build ## Build and deploy to Cloud Run
	gcloud run deploy $(SERVICE_NAME) \
		--image $(IMAGE) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--memory $(MEMORY) \
		--cpu $(CPU) \
		--timeout $(TIMEOUT) \
		--session-affinity \
		--min-instances 0 \
		--max-instances 3 \
		--set-secrets="/app/client_secret.json=client-secret-json:latest,/app/.streamlit/secrets.toml=streamlit-secrets:latest"
	@echo ""
	@echo "── Deployed! ──────────────────────────────────────────"
	@echo "URL: $$(gcloud run services describe $(SERVICE_NAME) --region $(REGION) --format 'value(status.url)')"
	@echo ""
	@echo "IMPORTANT: Add this URL to Google OAuth authorized redirect URIs"
	@echo "and update .streamlit/secrets.toml redirect_uri, then run: make update-secrets"
	@echo "──────────────────────────────────────────────────────"

# ── Local ─────────────────────────────────────────────────────────────
.PHONY: run docker-run

run: ## Run locally with Streamlit
	streamlit run app.py

docker-build: ## Build Docker image locally
	docker build -t $(SERVICE_NAME) .

docker-run: docker-build ## Build and run locally in Docker
	docker run -p 8080:8080 \
		-v "$(PWD)/client_secret.json:/app/client_secret.json:ro" \
		-v "$(PWD)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro" \
		$(SERVICE_NAME)

# ── Monitoring ────────────────────────────────────────────────────────
.PHONY: logs status url

logs: ## Tail Cloud Run logs
	gcloud run services logs read $(SERVICE_NAME) --region $(REGION) --limit 50

status: ## Show service status
	gcloud run services describe $(SERVICE_NAME) --region $(REGION)

url: ## Print the service URL
	@gcloud run services describe $(SERVICE_NAME) --region $(REGION) --format 'value(status.url)'

# ── Cleanup ───────────────────────────────────────────────────────────
.PHONY: delete

delete: ## Delete the Cloud Run service
	gcloud run services delete $(SERVICE_NAME) --region $(REGION)

# ── Help ──────────────────────────────────────────────────────────────
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
