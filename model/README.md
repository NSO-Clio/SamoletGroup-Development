This is a microservice that may be used to easily upscale models.
You can use it with both dockerized and normal versions.

In both cases you should set the environment variable with path to model weights. If you run the module normally, set it with 

```
export SCORING_MODEL_PATH="/path_to_model" SCORING_MEAN_MEDIAN_IMPUTER_PATH="/path_to_mmimputer"
```
Alternatively you may create the .env file and specify your variables there.

For docker you should set both environment variables and mount the volume with weights.
```
docker run -e SCORING_MODEL_PATH="/weights/model" -e SCORING_MEAN_MEDIAN_IMPUTER_PATH="/weights/mmimputer" -v ./weights:/weights ...
```

Build the docker container with 
```
docker build -t aij_model .
```
Run with
```
docker run -e SCORING_MODEL_PATH="/weights/model" \
	-e SCORING_MEAN_MEDIAN_IMPUTER_PATH="/weights/mmimputer" \
	-v ./weights:/weights \
	-p 8000:8000 \
	aij_model
```
