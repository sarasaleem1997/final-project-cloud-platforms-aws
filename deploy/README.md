# Deployment

## SageMaker deployment

Run the deployment script:

```bash
uv run python deploy/deploy_sagemaker.py \
  --bucket vaultech-models-999390550986 \
  --region eu-west-1 \
  --endpoint-name vaultech-bath-predictor \
  --model-package-group vaultech-bath-predictor-group
```

## Resource names

| Resource | Name |
|---|---|
| S3 bucket | vaultech-models-999390550986 |
| Model Package Group | vaultech-bath-predictor-group |
| Endpoint name | vaultech-bath-predictor |
| AWS region | eu-west-1 |

## Validate

```bash
export SAGEMAKER_MODEL_PACKAGE_GROUP="vaultech-bath-predictor-group"
export SAGEMAKER_ENDPOINT_NAME="vaultech-bath-predictor"
export AWS_DEFAULT_REGION="eu-west-1"
uv run pytest tests/test_sagemaker.py -v
```
