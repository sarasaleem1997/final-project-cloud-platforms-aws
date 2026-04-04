"""
SageMaker deployment script — packages, registers, and deploys the XGBoost model.

Usage:
    uv run python deploy/deploy_sagemaker.py \
      --bucket your-bucket-name \
      --region eu-west-1 \
      --endpoint-name your-endpoint-name \
      --model-package-group your-group-name
"""

import argparse
import io
import json
import shutil
import tarfile
import time
from pathlib import Path

import boto3
import sagemaker
from sagemaker import image_uris


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_DIR / "xgboost_bath_predictor.json"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


def package_model(model_path: Path, output_dir: Path) -> Path:
    """Package the XGBoost model as a .tar.gz archive for SageMaker.

    SageMaker's built-in XGBoost container expects a file named
    'xgboost-model' at the root of the archive.

    XGBoost 3.2+ stores base_score in array notation ('[58.6]') in the JSON.
    The SageMaker 3.0-5 container expects a scalar string ('58.6'). This
    function patches that field before archiving so the container loads the
    model with the correct intercept.

    Args:
        model_path: Path to the trained model JSON file.
        output_dir: Directory where the .tar.gz will be created.

    Returns:
        Path to the created .tar.gz file.
    """
    # Load and patch the model JSON for container compatibility
    with open(model_path) as f:
        model_json = json.load(f)

    lmp = model_json.get("learner", {}).get("learner_model_param", {})
    raw_bs = lmp.get("base_score", "")
    # XGBoost 3.2 stores base_score as '[value]' — strip brackets for 3.0 compat
    if raw_bs.startswith("[") and raw_bs.endswith("]"):
        lmp["base_score"] = raw_bs[1:-1]

    patched_bytes = json.dumps(model_json).encode("utf-8")

    tar_path = output_dir / "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        # SageMaker XGBoost container expects the model at root as 'xgboost-model'
        info = tarfile.TarInfo(name="xgboost-model")
        info.size = len(patched_bytes)
        tar.addfile(info, io.BytesIO(patched_bytes))

    return tar_path


def upload_to_s3(local_path: Path, bucket: str, key: str) -> str:
    """Upload a local file to S3.

    Args:
        local_path: Path to the local file.
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        Full S3 URI (s3://bucket/key).
    """
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def register_model(
    s3_model_uri: str,
    model_package_group_name: str,
    region: str,
    metrics: dict,
) -> str:
    """Register the model in SageMaker Model Registry.

    Creates the Model Package Group if it doesn't exist, then registers
    a new Model Package version with the XGBoost container image,
    the S3 model artifact, and evaluation metrics.

    Args:
        s3_model_uri: S3 URI of the packaged model (.tar.gz).
        model_package_group_name: Name for the Model Package Group.
        region: AWS region.
        metrics: Dict with 'rmse', 'mae', 'r2' keys.

    Returns:
        The Model Package ARN.
    """
    sm = boto3.client("sagemaker", region_name=region)

    # Create Model Package Group if it doesn't exist
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription="VaultTech forging line bath time predictor",
        )
        print(f"  Created Model Package Group: {model_package_group_name}")
    except sm.exceptions.ClientError as e:
        if "already exists" in str(e) or "ConflictException" in str(e):
            print(f"  Model Package Group already exists: {model_package_group_name}")
        else:
            raise

    # Get the XGBoost container image URI for eu-west-1
    # Using 3.0-5 container (compatible with XGBoost 3.x models trained locally)
    image_uri = image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="3.0-5",
    )

    # Register the model package
    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription="XGBoost 3.x bath time predictor — MAE=0.92s, R2=0.69",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": s3_model_uri,
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "ContentType": "application/json",
                    "S3Uri": s3_model_uri,  # placeholder — metrics attached below
                }
            }
        },
        ModelApprovalStatus="Approved",
        CustomerMetadataProperties={
            "rmse": str(metrics["rmse"]),
            "mae": str(metrics["mae"]),
            "r2": str(metrics["r2"]),
        },
    )

    return response["ModelPackageArn"]


def deploy_endpoint(
    model_package_arn: str,
    endpoint_name: str,
    region: str,
    instance_type: str = "ml.t2.medium",
) -> str:
    """Deploy a real-time SageMaker endpoint from a registered Model Package.

    Creates a SageMaker Model, Endpoint Configuration, and Endpoint.
    Waits until the endpoint status is 'InService'.

    Args:
        model_package_arn: ARN of the registered Model Package.
        endpoint_name: Name for the endpoint.
        region: AWS region.
        instance_type: EC2 instance type for the endpoint.

    Returns:
        The endpoint name.
    """
    sm = boto3.client("sagemaker", region_name=region)
    role = sagemaker.get_execution_role() if _is_sagemaker_env() else _get_iam_role(region)

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"

    # Create SageMaker Model from the registered package
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        Containers=[{"ModelPackageName": model_package_arn}],
    )

    # Create Endpoint Configuration
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "default",
                "ModelName": model_name,
                "InstanceType": instance_type,
                "InitialInstanceCount": 1,
            }
        ],
    )

    # Create Endpoint
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )

    # Wait for endpoint to be InService
    print(f"  Waiting for endpoint to be InService (this takes ~5 minutes)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 30},
    )

    return endpoint_name


def test_endpoint(endpoint_name: str, region: str) -> dict:
    """Test the deployed endpoint with sample pieces.

    Invokes the endpoint with representative inputs and compares
    the predictions against expected ranges.

    Args:
        endpoint_name: Name of the deployed endpoint.
        region: AWS region.

    Returns:
        Dict with test results and predictions.
    """
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    # Sample pieces: die_matrix, lifetime_2nd_strike_s, oee_cycle_time_s
    test_cases = [
        {"die_matrix": 5052, "lifetime_2nd_strike_s": 18.3, "oee_cycle_time_s": 13.5},
        {"die_matrix": 4974, "lifetime_2nd_strike_s": 17.4, "oee_cycle_time_s": 13.9},
        {"die_matrix": 5090, "lifetime_2nd_strike_s": 17.7, "oee_cycle_time_s": 14.0},
        {"die_matrix": 5091, "lifetime_2nd_strike_s": 18.5, "oee_cycle_time_s": 13.8},
        {"die_matrix": 5052, "lifetime_2nd_strike_s": 30.0, "oee_cycle_time_s": 13.5},  # slow piece
    ]

    results = []
    for case in test_cases:
        payload = f"{case['die_matrix']},{case['lifetime_2nd_strike_s']},{case['oee_cycle_time_s']}"
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=payload,
        )
        prediction = float(response["Body"].read().decode("utf-8").strip())
        results.append({
            "input": case,
            "predicted_bath_time_s": round(prediction, 3),
            "in_range": 40 < prediction < 80,
        })

    return {
        "endpoint": endpoint_name,
        "predictions": results,
        "all_in_range": all(r["in_range"] for r in results),
        "slow_piece_higher": results[-1]["predicted_bath_time_s"] > results[0]["predicted_bath_time_s"],
    }


def _is_sagemaker_env() -> bool:
    """Check if running inside a SageMaker environment (notebook instance or job)."""
    import os
    from pathlib import Path
    # SM_CURRENT_HOST is set by SageMaker training/processing jobs
    # /opt/ml/ exists on SageMaker notebook instances and jobs
    return "SM_CURRENT_HOST" in os.environ or Path("/opt/ml/").exists()


def _get_iam_role(region: str) -> str:
    """Get SageMaker execution role from IAM, preferring dedicated execution roles."""
    iam = boto3.client("iam", region_name=region)
    roles = iam.list_roles()["Roles"]
    # Prefer non-service-linked roles (service-linked roles are in aws-reserved path)
    candidates = [
        r for r in roles
        if ("SageMaker" in r["RoleName"] or "sagemaker" in r["RoleName"].lower())
        and "aws-reserved" not in r["Arn"]
        and "AWSServiceRole" not in r["RoleName"]
    ]
    if candidates:
        return candidates[0]["Arn"]
    raise RuntimeError("No SageMaker IAM role found. Create one in the AWS console.")


def main():
    parser = argparse.ArgumentParser(description="Deploy XGBoost model to SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket for model artifact")
    parser.add_argument("--region", default="eu-west-1", help="AWS region")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    parser.add_argument("--model-package-group", required=True, help="Model Package Group name")
    args = parser.parse_args()

    # Load model metadata for metrics
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    print("=" * 60)
    print("SageMaker Deployment Pipeline")
    print("=" * 60)

    # Step 1: Package
    print("\n[1/5] Packaging model artifact...")
    tar_path = package_model(MODEL_FILE, MODEL_DIR)
    print(f"  Created: {tar_path}")

    # Step 2: Upload to S3
    print("\n[2/5] Uploading to S3...")
    s3_key = "models/xgboost-bath-predictor/model.tar.gz"
    s3_uri = upload_to_s3(tar_path, args.bucket, s3_key)
    print(f"  Uploaded: {s3_uri}")

    # Step 3: Register in Model Registry
    print("\n[3/5] Registering in Model Registry...")
    model_package_arn = register_model(
        s3_uri, args.model_package_group, args.region, metadata["metrics"]
    )
    print(f"  Registered: {model_package_arn}")

    # Step 4: Deploy endpoint
    print("\n[4/5] Deploying endpoint...")
    endpoint = deploy_endpoint(model_package_arn, args.endpoint_name, args.region)
    print(f"  Endpoint live: {endpoint}")

    # Step 5: Test
    print("\n[5/5] Testing endpoint...")
    results = test_endpoint(args.endpoint_name, args.region)
    print(f"  Results: {json.dumps(results, indent=2)}")

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"  Endpoint:       {args.endpoint_name}")
    print(f"  Model Package:  {model_package_arn}")
    print(f"  S3 artifact:    {s3_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
